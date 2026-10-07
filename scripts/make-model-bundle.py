#!/usr/bin/env python3
"""Assemble a WeMM-Embedding-2B CoreML bundle in the release layout.

Layout (text + image + video):
  spec.yaml
  tokenizer.json, tokenizer_config.json
  embed_table.fp16.npy
  vision_abs_pos.fp32.npy
  vision_tower.mlmodelc/            two-frame vision tower (+ manifest.txt)
  chunks/chunk{0..5}.mlmodelc/      position-input decoder chunks (+ manifest.txt)

Pass --text-only to build a text-only bundle (no tower or position table; then
chunks/ must be the plain decoder).

Modes:
  --tag v1.0.0   Hugging Face upload folder. spec.yaml pins `revision: <tag>`;
                 create that tag on the repo right after uploading.
  --install      Verified local install for testing (adds manifest.yaml and a
                 content-derived pseudo commit as the revision).

Each model directory gets an inner manifest.txt (`path size sha256` lines,
sorted by UTF-8 path, LF). Only the standard library is used.

Example:
  python3 scripts/make-model-bundle.py --install \\
    --tokenizer-dir <upstream WeMM-Embedding-2B> --embed-table <embed_table.fp16.npy> \\
    --positions <vision_abs_pos.fp32.npy> --tower <vision_tower.mlmodelc> \\
    --chunks-dir <dir> --chunk-pattern 'wemm_chunk{c}_of6_seq512_extrope.mlmodelc' \\
    --out <model-root>/wemm-embedding-2b
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = "maolon/WeMM-Embedding-2B-CoreML-ANE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, dest: Path) -> None:
    """APFS clone when possible (instant, no extra space), else a plain copy."""
    if subprocess.run(["cp", "-cR", str(source), str(dest)], capture_output=True).returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(source, dest)


def inner_manifest(model_dir: Path) -> bytes:
    entries = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"{path} is a symlink")
        if path.is_file():
            relative = path.relative_to(model_dir).as_posix()
            if relative == "manifest.txt":
                raise SystemExit(f"{model_dir} already contains manifest.txt")
            entries.append((relative, path.stat().st_size, sha256_file(path)))
    entries.sort(key=lambda entry: entry[0].encode("utf-8"))
    return "".join(f"{p} {s} {h}\n" for p, s, h in entries).encode("utf-8")


def add_model_dir(source: Path, out: Path, relative: str, files: list) -> None:
    if not source.is_dir():
        raise SystemExit(f"missing model directory {source}")
    dest = out / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(source, dest)
    data = inner_manifest(dest)
    (dest / "manifest.txt").write_bytes(data)
    files.append({"path": f"{relative}/manifest.txt", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})


def add_file(source: Path, out: Path, name: str, files: list) -> None:
    if not source.is_file():
        raise SystemExit(f"missing file {source}")
    shutil.copy2(source, out / name)
    files.append({"path": name, "size": (out / name).stat().st_size, "sha256": sha256_file(out / name)})


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--embed-table", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--chunk-pattern", default="chunk{c}.mlmodelc")
    parser.add_argument("--tower", type=Path, help="two-frame vision_tower .mlmodelc")
    parser.add_argument("--positions", type=Path, help="vision_abs_pos.fp32.npy")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--id", default="wemm-embedding-2b-ane", help="must match the app default model_id for a zero-config install")
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tag", help="release tag the spec pins, e.g. v1.0.0")
    mode.add_argument("--install", action="store_true")
    args = parser.parse_args()

    if not args.text_only and not (args.tower and args.positions):
        raise SystemExit("a multimodal bundle needs --tower and --positions (or pass --text-only)")
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"destination {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    files: list = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        add_file(args.tokenizer_dir / name, out, name, files)
    add_file(args.embed_table, out, "embed_table.fp16.npy", files)
    for c in range(6):
        add_model_dir(args.chunks_dir / args.chunk_pattern.format(c=c), out, f"chunks/chunk{c}.mlmodelc", files)
    if not args.text_only:
        add_model_dir(args.tower, out, "vision_tower.mlmodelc", files)
        add_file(args.positions, out, "vision_abs_pos.fp32.npy", files)
    files.sort(key=lambda entry: entry["path"].encode("utf-8"))

    # Install mode pins a content-derived 40-hex pseudo commit; release mode a tag.
    pseudo_commit = hashlib.sha1("".join(f["sha256"] for f in files).encode()).hexdigest()
    revision = pseudo_commit if args.install else args.tag
    spec = "\n".join([
        "spec_version: 1",
        "model:",
        f"  id: {args.id}",
        "  dim: 2048",
        "  max_seq: 512",
        "  normalize: l2",
        "source:",
        f"  repo: {REPO}",
        f"  revision: {revision}",
        "  endpoint: https://huggingface.co",
        "files:",
        *[f"  - {{path: {f['path']}, sha256: {f['sha256']}, size: {f['size']}}}" for f in files],
        "runtime:",
        "  compute_units: cpu_and_ne",
        "  pad_side: right",
        "  mask_dtype: fp32",
        "",
    ])
    (out / "spec.yaml").write_text(spec)
    if args.install:
        lines = []
        for f in files:
            lines += [f"    - path: {f['path']}", f"      sha256: {f['sha256']}", f"      size: {f['size']}"]
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = "\n".join([
            "manifest_version: 1",
            f"spec_digest: {hashlib.sha256(spec.encode()).hexdigest()}",
            f"resolved_commit: {pseudo_commit}",
            "files:", *lines,
            "generator: embed-ane/0.1.0",
            f"created_at: {yaml_quote(created)}",
            "spec_yaml: " + yaml_quote(spec),
            "",
        ])
        (out / "manifest.yaml").write_text(manifest)
    kind = "text-only" if args.text_only else "multimodal"
    print(f"{kind} bundle at {out}: {len(files)} spec entries, revision {revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
