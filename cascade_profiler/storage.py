"""Optional finished-run S3 uploads; authentication stays in the caller's environment."""

import json
from urllib.parse import urlsplit


def s3_destination(options, run_name):
    from cascade_profiler.legacy import S3Dest

    defaults = {"endpoint": "", "prefix": "accelsim-traces", "region": "auto", "addressing_style": "auto"}
    if not isinstance(options, dict) or set(options) - {*defaults, "bucket"}:
        raise ValueError("s3 accepts only endpoint, bucket, prefix, region, addressing_style; credentials belong in env")
    config = {**defaults, **options}
    if not all(isinstance(v, str) for v in config.values()) or not config.get("bucket"):
        raise ValueError("s3 requires a nonempty bucket and string destination fields")
    endpoint = urlsplit(config["endpoint"])
    if config["endpoint"] and (endpoint.scheme not in {"http", "https"} or not endpoint.netloc or
                               endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("s3 endpoint must be an HTTP(S) URL without credentials, query or fragment")
    if config["addressing_style"] not in {"auto", "path", "virtual"}:
        raise ValueError("s3 addressing_style must be auto, path or virtual")
    config["prefix"] = config["prefix"].strip("/")
    destination = S3Dest(run_name, **config)
    return destination, {**config, "uri": destination.uri()}


def upload_run(run_dir, destination, manifest):
    path = run_dir / "run.json"
    try:
        for file in sorted(run_dir.rglob("*")):
            if file.is_file() and file != path:
                destination.upload(file, file.relative_to(run_dir).as_posix())
        manifest["s3_upload_status"] = "completed"
        path.write_text(json.dumps(manifest, indent=2))
        destination.upload(path, "run.json")
    except Exception:
        manifest["s3_upload_status"] = "failed"
        path.write_text(json.dumps(manifest, indent=2))
        raise
