import os
import re
import subprocess
from pathlib import Path

STEP_RE = re.compile(r"^vllm_step_(\d+)$")
STEP_KIND_RE = re.compile(r"^(prefill|decode|mixed|empty|unknown)_(\d+)$")


def _nsys_base_cmd(gpu_metrics: bool = False, capture_range: bool = False,
                   cuda_events: bool = True, nsys_bin: str = "nsys") -> list[str]:
    """nsys profile flags; optional ones are only added if this nsys version knows them."""
    help_text = subprocess.run([nsys_bin, "profile", "--help"], capture_output=True, text=True).stdout
    cmd = [
        nsys_bin, "profile",
        "--trace=cuda,nvtx",
        # No CPU sampling / context-switch tracing: needs perf, unavailable in gVisor, not needed.
        "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true", "--export=sqlite",
    ]
    # One CUPTI record per kernel *node* of a CUDA graph (default: one per graph launch).
    if "--cuda-graph-trace" in help_text:
        cmd.append("--cuda-graph-trace=node")
    # cudaEventRecord/cudaStreamWaitEvent etc. as activity records (synchronization table).
    if "--cuda-event-trace" in help_text:
        cmd.append(f"--cuda-event-trace={str(cuda_events).lower()}")
    if capture_range:
        cmd.append("--capture-range=cudaProfilerApi")
        # Default "stop-shutdown" SIGTERMs the app at cudaProfilerStop (exit 143).
        cmd.append("--capture-range-end=stop")
    if gpu_metrics:
        flag = "--gpu-metrics-devices" if "--gpu-metrics-devices" in help_text else "--gpu-metrics-device"
        cmd.append(f"{flag}=all")
    return cmd


def _nsys_env(extra: dict | None = None, accelsim_root: str | None = None) -> dict:
    env = {
        **os.environ,
        # Same process layout as the traced run (run.sh sets these); no CUDA_INJECTION64_PATH,
        # so the NVBit hooks are no-ops and nsys owns CUDA injection.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        **(extra or {}),
    }
    if accelsim_root is not None:
        root = Path(accelsim_root) / "util/tracer_nvbit"
        env["PYTHONPATH"] = f"{root}/others:{root}/tracer_tool:{env.get('PYTHONPATH', '')}"
    env.pop("CUDA_INJECTION64_PATH", None)
    return {k: v for k, v in env.items() if k in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
            or not (k.upper().startswith("MODAL_") or
                    re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|ACCESS_KEY", k, re.I))}


def _scrub_nsys(work: Path, keep_nsys_rep: bool = False):
    import sqlite3

    db = work / "profile.sqlite"
    if db.exists():
        with sqlite3.connect(db) as con:
            if _has_table(con, "META_DATA_CAPTURE"):
                cols = [r[1] for r in con.execute('PRAGMA table_info("META_DATA_CAPTURE")')]
                quoted = ['"' + c.replace('"', '""') + '"' for c in cols]
                if quoted:
                    con.execute('DELETE FROM META_DATA_CAPTURE WHERE ' + ' OR '.join(
                        f"(typeof({c})='text' AND {c} LIKE '%ENVIRONMENT_VARIABLE%')"
                        for c in quoted))
            con.commit()
            con.execute("VACUUM")
    if keep_nsys_rep:
        print("WARNING: .nsys-rep contains the unscrubbed environment (including HF tokens)")
    else:
        (work / "profile.nsys-rep").unlink(missing_ok=True)


def _has_table(con, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None


def _cols(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _enum(con, table: str) -> dict:
    """nsys ENUM_* table as {id: label}."""
    if not _has_table(con, table):
        return {}
    c = _cols(con, table)
    label = "label" if "label" in c else "name"
    return {i: v for i, v in con.execute(f"SELECT id, {label} FROM {table}")}


def _rt_join(con, alias: str) -> tuple[str, str]:
    """(select columns, join clause) for the runtime API call that issued `alias`'s activity."""
    if not _has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return "NULL, NULL, NULL, NULL", ""
    return ("r.start, r.end, rn.value, r.globalTid",
            f"LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = {alias}.correlationId "
            f"AND r.globalTid / 16777216 = {alias}.globalPid / 16777216 "
            "LEFT JOIN StringIds rn ON rn.id = r.nameId")


def _pid_tid(global_tid) -> tuple:
    if global_tid is None:
        return "", ""
    return (global_tid >> 24) & 0xFFFFFF, global_tid & 0xFFFFFF


def _nsys_kernels(sqlite_path: Path) -> list[dict]:
    """All GPU kernels from an nsys sqlite export, in host launch order, with their API call."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    kcols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "mangledName" if "mangledName" in kcols else "demangledName"
    rt_sel, rt_join = _rt_join(con, "k")
    graph_node = "k.graphNodeId" if "graphNodeId" in kcols else "NULL"
    # Launch order = time of the host-side launch API call (what the tracer numbers by);
    # fall back to GPU start time if the API call wasn't captured. Kernels replayed from one
    # cudaGraphLaunch share its API call and are ordered by GPU start.
    launch_ts = "COALESCE(r.start, k.start)" if rt_join else "k.start"
    q = f"""
      SELECT k.start, k.end, k.streamId, k.deviceId, k.correlationId, {graph_node},
             k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
             k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory,
             sm.value, ss.value, {rt_sel}
      FROM CUPTI_ACTIVITY_KIND_KERNEL k
      JOIN StringIds sm ON sm.id = k.{name_col}
      JOIN StringIds ss ON ss.id = k.shortName
      {rt_join}
      ORDER BY {launch_ts}, k.start
    """
    out = []
    for (start, end, stream, dev, corr, gnode, gx, gy, gz, bx, by, bz, regs, smem, mangled, short,
         api_start, api_end, api_name, gtid) in con.execute(q):
        pid, tid = _pid_tid(gtid)
        out.append({
            "start": start, "end": end, "stream_id": stream, "device_id": dev,
            "correlation_id": corr, "graph_node_id": "" if gnode is None else gnode,
            "grid": f"{gx}x{gy}x{gz}", "block": f"{bx}x{by}x{bz}",
            "regs": regs, "smem": smem, "mangled": mangled, "short": short,
            "api_start": api_start, "api_end": api_end, "api_name": api_name or "",
            "pid": pid, "tid": tid,
        })
    return out


def _nsys_steps(sqlite_path: Path) -> dict[int, dict]:
    """vllm_step_<i> NVTX ranges (and their nested kind range) -> {i: {start, end, kind}}."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    if not _has_table(con, "NVTX_EVENTS"):
        return {}
    text = "COALESCE(n.text, s.value)" if "textId" in _cols(con, "NVTX_EVENTS") else "n.text"
    join = "LEFT JOIN StringIds s ON s.id = n.textId" if "textId" in _cols(con, "NVTX_EVENTS") else ""
    steps, kinds = {}, {}
    for start, end, name in con.execute(
            f"SELECT n.start, n.end, {text} FROM NVTX_EVENTS n {join} WHERE n.end IS NOT NULL"):
        if not name:
            continue
        if m := STEP_RE.match(name):
            steps[int(m.group(1))] = {"start": start, "end": end}
        elif m := STEP_KIND_RE.match(name):
            kinds[int(m.group(2))] = m.group(1)
        elif name == "vllm_init":
            steps.setdefault("init", {"start": start, "end": end})
    init = steps.pop("init", None)
    for i, st in steps.items():
        st["kind"] = kinds.get(i, "unknown")
    if init:
        steps[-1] = {**init, "kind": "init"}
    return steps


def _step_of(ts, steps: dict[int, dict]):
    if ts is None:
        return ""
    for i, st in steps.items():
        if i >= 0 and st["start"] <= ts <= st["end"]:
            return i
    init = steps.get(-1)
    return "init" if init and init["start"] <= ts <= init["end"] else ""


def _assign_steps(kernels: list[dict], steps: dict[int, dict]):
    for k in kernels:
        k["step"] = _step_of(k["api_start"] if k["api_start"] is not None else k["start"], steps)
        k["step_kind"] = steps[k["step"]]["kind"] if isinstance(k["step"], int) else k["step"]


def _nsys_events(sqlite_path: Path, t0: int, steps: dict[int, dict]) -> list[dict]:
    """Kernels, memcpys, memsets and synchronizations in GPU time order, with their API call."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    copy_kind = _enum(con, "ENUM_CUDA_MEMCPY_OPER")
    mem_kind = _enum(con, "ENUM_CUDA_MEM_KIND")
    sync_kind = _enum(con, "ENUM_CUPTI_SYNC_TYPE")
    out = []

    def add(etype, alias, table, extra_sel, fmt):
        if not _has_table(con, table):
            return
        rt_sel, rt_join = _rt_join(con, alias)
        cols = _cols(con, table)
        stream = f"{alias}.streamId" if "streamId" in cols else "NULL"
        dev = f"{alias}.deviceId" if "deviceId" in cols else "NULL"
        q = (f"SELECT {alias}.start, {alias}.end, {stream}, {dev}, {alias}.correlationId, "
             f"{rt_sel}, {extra_sel(cols)} FROM {table} {alias} {rt_join}")
        for row in con.execute(q):
            start, end, stream_id, dev_id, corr, api_start, api_end, api_name, gtid = row[:9]
            pid, tid = _pid_tid(gtid)
            ts = api_start if api_start is not None else start
            st = _step_of(ts, steps)
            out.append({
                "event_type": etype,
                "start_ns": start - t0, "end_ns": end - t0, "duration_ns": end - start,
                "abs_start_ns": start, "stream_id": stream_id, "device_id": dev_id,
                "correlation_id": corr, "api_name": api_name or "",
                "api_start_ns": "" if api_start is None else api_start - t0,
                "api_end_ns": "" if api_end is None else api_end - t0,
                "pid": pid, "tid": tid, "step": st,
                "step_kind": steps[st]["kind"] if isinstance(st, int) else st,
                "name": "", "bytes": "", "copy_kind": "", "src_kind": "", "dst_kind": "",
                "memset_value": "", "sync_type": "", "event_id": "",
                **fmt(row[9:]),
            })

    def col(cols, c, alias):
        return f"{alias}.{c}" if c in cols else "NULL"

    add("kernel", "k", "CUPTI_ACTIVITY_KIND_KERNEL",
        lambda c: "(SELECT value FROM StringIds WHERE id = k.shortName)",
        lambda r: {"name": r[0]})
    add("memcpy", "m", "CUPTI_ACTIVITY_KIND_MEMCPY",
        lambda c: f"m.bytes, m.copyKind, {col(c, 'srcKind', 'm')}, {col(c, 'dstKind', 'm')}",
        lambda r: {"bytes": r[0], "copy_kind": copy_kind.get(r[1], r[1]),
                   "src_kind": mem_kind.get(r[2], r[2] if r[2] is not None else ""),
                   "dst_kind": mem_kind.get(r[3], r[3] if r[3] is not None else ""),
                   "name": f"memcpy {copy_kind.get(r[1], r[1])}"})
    add("memset", "s", "CUPTI_ACTIVITY_KIND_MEMSET",
        lambda c: f"s.bytes, {'s.value' if 'value' in c else 'NULL'}",
        lambda r: {"bytes": r[0], "memset_value": "" if r[1] is None else r[1], "name": "memset"})
    add("sync", "y", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        lambda c: f"y.syncType, {'y.eventId' if 'eventId' in c else 'NULL'}",
        lambda r: {"sync_type": sync_kind.get(r[0], r[0]),
                   "event_id": "" if r[1] is None else r[1],
                   "name": f"sync {sync_kind.get(r[0], r[0])}"})
    out.sort(key=lambda e: (e["abs_start_ns"], e["end_ns"]))
    return out


def _nsys_api_calls(sqlite_path: Path, t0: int, steps: dict[int, dict]) -> list[dict]:
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    if not _has_table(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    out = []
    rv_col = "r.returnValue" if "returnValue" in _cols(con, "CUPTI_ACTIVITY_KIND_RUNTIME") else "NULL"
    for start, end, corr, name, gtid, rv in con.execute(
            f"SELECT r.start, r.end, r.correlationId, s.value, r.globalTid, {rv_col} "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON s.id = r.nameId "
            "ORDER BY r.start"):
        pid, tid = _pid_tid(gtid)
        st = _step_of(start, steps)
        out.append({"api_name": name, "start_ns": start - t0, "end_ns": end - t0,
                    "duration_ns": end - start, "abs_start_ns": start, "correlation_id": corr,
                    "return_value": rv, "pid": pid, "tid": tid, "step": st,
                    "step_kind": steps[st]["kind"] if isinstance(st, int) else st})
    return out


def _tracer_launches(stats_path: Path) -> list[dict]:
    """Every kernel launch the tracer saw (traced or not), in launch order."""
    out = []
    for line in stats_path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        m = re.match(r"kernel-(\d+)-", parts[0])
        out.append({
            "id": int(m.group(1)),
            "mangled": parts[1],
            "grid": f"{parts[2]}x{parts[3]}x{parts[4]}",
            "block": f"{parts[6]}x{parts[7]}x{parts[8]}",
            "traced": int(parts[10]) > 0,
        })
    return out


def _align(kernels: list[dict], traced: list[dict]) -> tuple[list[dict], dict]:
    """Match nsys kernels to tracer launch ids by mangled-name sequence.

    Matching is by name only: launch dims can legitimately differ between the traced and the
    profiled run (e.g. vLLM sizes the KV cache from free GPU memory, and NVBit takes some, so
    the KV-cache fill kernels get different grids). Differences are flagged per row instead.
    """
    import difflib

    a, b = [k["mangled"] for k in kernels], [t["mangled"] for t in traced]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    nsys_to_trace = {}
    for blk in sm.get_matching_blocks():
        for i in range(blk.size):
            nsys_to_trace[blk.a + i] = blk.b + i
    exact = a == b
    t0 = kernels[0]["start"] if kernels else 0

    def rel(ts):
        return "" if ts is None else ts - t0

    rows, prev_end = [], None
    for i, k in enumerate(kernels):
        t = traced[nsys_to_trace[i]] if i in nsys_to_trace else None
        rows.append({
            "launch_seq": i + 1,
            "trace_kernel_id": t["id"] if t else "",
            "traced": bool(t and t["traced"]),
            "launch_dims_match": "" if not t else (t["grid"], t["block"]) == (k["grid"], k["block"]),
            "step": k.get("step", ""), "step_kind": k.get("step_kind", ""),
            "start_ns": k["start"] - t0, "end_ns": k["end"] - t0,
            "duration_ns": k["end"] - k["start"],
            "gap_from_prev_end_ns": "" if prev_end is None else k["start"] - prev_end,
            "abs_start_ns": k["start"],
            "api_name": k.get("api_name", ""),
            "api_start_ns": rel(k.get("api_start")), "api_end_ns": rel(k.get("api_end")),
            "launch_to_start_ns": "" if k.get("api_start") is None else k["start"] - k["api_start"],
            "correlation_id": k.get("correlation_id", ""),
            "pid": k.get("pid", ""), "tid": k.get("tid", ""), "device_id": k.get("device_id", ""),
            "stream_id": k["stream_id"], "graph_node_id": k.get("graph_node_id", ""),
            "grid": k["grid"], "block": k["block"],
            "registers_per_thread": k["regs"], "shared_mem_bytes": k["smem"],
            "short_name": k["short"], "mangled_name": k["mangled"],
        })
        prev_end = k["end"]
    # Traced kernels that found no nsys partner are listed so nothing is silently dropped.
    matched_ids = {r["trace_kernel_id"] for r in rows if r["trace_kernel_id"] != ""}
    blank = {c: "" for c in rows[0]} if rows else {}
    for t in traced:
        if t["traced"] and t["id"] not in matched_ids:
            rows.append(blank | {"trace_kernel_id": t["id"], "traced": True, "grid": t["grid"],
                                 "block": t["block"], "mangled_name": t["mangled"]})
    alignment = {
        "exact_sequence_match": exact,
        "matched": len(nsys_to_trace),
        "nsys_unmatched": len(kernels) - len(nsys_to_trace),
        "tracer_unmatched": len(traced) - len(nsys_to_trace),
        "launch_dims_mismatched": sum(1 for r in rows if r["launch_dims_match"] is False),
        "traced_launch_dims_mismatched": sum(
            1 for r in rows if r["traced"] and r["launch_dims_match"] is False),
    }
    return rows, alignment


def _per_step(rows: list[dict], steps: dict[int, dict], step_log: list[dict],
              trace_step_log: list[dict], sqlite_path: Path) -> list[dict]:
    """Per engine step: timing from nsys + kernel matching against the tracer's step ranges."""
    import sqlite3

    con = sqlite3.connect(sqlite_path)
    busy_iv: dict[int, list] = {}
    memcpy: dict[int, list] = {}
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET"):
        if not _has_table(con, table):
            continue
        sel = "start, end" + (", bytes" if table.endswith("MEMCPY") else "")
        for r in con.execute(f"SELECT {sel} FROM {table}"):
            st = _step_of(r[0], steps)
            if isinstance(st, int) and st >= 0:
                busy_iv.setdefault(st, []).append((r[0], r[1]))
                if table.endswith("MEMCPY"):
                    memcpy.setdefault(st, []).append(r[2])
    log = {s["step"]: s for s in step_log}
    tlog = {s["step"]: s for s in trace_step_log}
    out = []
    for i in sorted(s for s in steps if s >= 0):
        st = steps[i]
        ks = [r for r in rows if r["step"] == i]
        iv = sorted(busy_iv.get(i, []))
        busy, cur_s, cur_e = 0, None, None
        for s0, e0 in iv:
            if cur_e is None or s0 > cur_e:
                busy += 0 if cur_e is None else cur_e - cur_s
                cur_s, cur_e = s0, e0
            else:
                cur_e = max(cur_e, e0)
        busy += 0 if cur_e is None else cur_e - cur_s
        span = st["end"] - st["start"]
        tr = tlog.get(i, {})
        lo, hi = tr.get("first_kernel_id"), tr.get("end_kernel_id")
        in_range = (lambda kid: kid != "" and lo is not None and hi is not None and lo <= kid < hi)
        matched = sum(1 for r in ks if in_range(r["trace_kernel_id"]))
        out.append({
            "step": i, "kind": st["kind"],
            "num_scheduled_tokens": log.get(i, {}).get("num_scheduled_tokens", ""),
            "num_reqs": log.get(i, {}).get("num_reqs", ""),
            "nvtx_start_ns": st["start"], "nvtx_end_ns": st["end"], "span_ns": span,
            "gpu_busy_ns": busy, "gpu_idle_ns": span - busy,
            "kernel_sum_ns": sum(r["duration_ns"] for r in ks),
            "first_launch_to_first_start_ns": ks[0]["launch_to_start_ns"] if ks else "",
            "nsys_kernels": len(ks),
            "memcpys": len(memcpy.get(i, [])), "memcpy_bytes": sum(memcpy.get(i, [])),
            "tracer_kernels": "" if lo is None or hi is None else hi - lo,
            "tracer_first_kernel_id": "" if lo is None else lo,
            "tracer_end_kernel_id": "" if hi is None else hi,
            "tracer_traced": tr.get("traced", ""),
            "matched_in_step": matched,
            "exact_step_match": bool(ks) and matched == len(ks) == (hi - lo if lo and hi else -1),
            "traced_matched": sum(1 for r in ks if r["traced"] and in_range(r["trace_kernel_id"])),
        })
    return out


def _nsys_nvtx(db: Path) -> list[dict]:
    import sqlite3

    with sqlite3.connect(db) as con:
        if not _has_table(con, "NVTX_EVENTS"):
            return []
        cols = _cols(con, "NVTX_EVENTS")
        text = "COALESCE(n.text,s.value)" if "textId" in cols else "n.text"
        join = "LEFT JOIN StringIds s ON s.id=n.textId" if "textId" in cols else ""
        domain = "n.domainId" if "domainId" in cols else "NULL"
        tid = "n.globalTid" if "globalTid" in cols else "NULL"
        rows = []
        for a, b, name, dom, gtid in con.execute(
                f"SELECT n.start,n.end,{text},{domain},{tid} FROM NVTX_EVENTS n {join}"):
            pid, thread = _pid_tid(gtid)
            rows.append({"start": a, "end": b, "text": name, "domain": dom,
                         "pid": pid, "tid": thread})
        return rows
