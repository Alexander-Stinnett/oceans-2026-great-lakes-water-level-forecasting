from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oceans-glwl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce-paper", help="Regenerate all manuscript tables and figures")
    reproduce.add_argument("--root", type=Path)
    reproduce.add_argument("--check-only", action="store_true")
    verify = subparsers.add_parser("verify-artifacts", help="Verify frozen artifact hashes")
    verify.add_argument("--manifest", type=Path)
    device = subparsers.add_parser("device", help="Resolve a portable training device")
    device.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reproduce-paper":
        from oceans_glwl.paper import reproduce_paper

        print(json.dumps(reproduce_paper(root=args.root, write_outputs=not args.check_only), indent=2, default=str))
        return 0
    if args.command == "verify-artifacts":
        from oceans_glwl.artifacts.manifests import verify_manifest

        failures = verify_manifest(args.manifest)
        print(json.dumps({"status": "passed" if not failures else "failed", "failures": failures}, indent=2))
        return 0 if not failures else 1
    if args.command == "device":
        from oceans_glwl.training.devices import resolve_device

        print(json.dumps(resolve_device(args.device).__dict__, indent=2))
        return 0
    raise AssertionError(args.command)

