"""Upload a local trace directory to main's S3 endpoint the way the profiler does.

Same client settings as Alessio's S3Dest (unsigned, path-style, checksums only when required,
128 MiB multipart by default), same layout (<bucket>/<prefix>/<run_name>/...), and metadata.json
last with `status` and `files`.

    uv run python scripts/upload_trace.py ./my-run --run-name my-run \
        --endpoint http://localhost:9000 --meta trace_gpu_arch=sm_70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("src", type=Path, help="directory holding traces/ (and optionally logs)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--endpoint", default="http://localhost:9000")
    p.add_argument("--bucket", default="cascade")
    p.add_argument("--prefix", default="accelsim-traces")
    p.add_argument("--part-mib", type=int, default=128)
    p.add_argument("--status", default="ok", choices=["ok", "failed"])
    p.add_argument(
        "--meta",
        action="append",
        default=[],
        help="extra metadata.json key=value (JSON if it starts with [ or {)",
    )
    args = p.parse_args()

    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        region_name="auto",
        config=Config(
            signature_version=UNSIGNED,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "path"},
        ),
    )
    part = args.part_mib * 1024 * 1024
    transfer = TransferConfig(multipart_threshold=part, multipart_chunksize=part)
    base = f"{args.prefix.strip('/')}/{args.run_name}"

    files = []
    for path in sorted(q for q in args.src.rglob("*") if q.is_file() and q.name != "metadata.json"):
        rel = path.relative_to(args.src).as_posix()
        client.upload_file(str(path), args.bucket, f"{base}/{rel}", Config=transfer)
        files.append({"key": f"{base}/{rel}", "bytes": path.stat().st_size})
        print(f"uploaded {rel} ({path.stat().st_size} bytes)")

    meta = {"status": args.status, "files": files}
    for kv in args.meta:
        key, _, value = kv.partition("=")
        meta[key] = json.loads(value) if value[:1] in "[{" else value
    client.put_object(
        Bucket=args.bucket, Key=f"{base}/metadata.json", Body=json.dumps(meta, indent=2).encode()
    )
    print(f"uploaded metadata.json ({len(files)} files listed) to s3://{args.bucket}/{base}/")


if __name__ == "__main__":
    main()
