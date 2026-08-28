import json

from oceans_glwl.artifacts.manifests import verify_manifest

if __name__ == "__main__":
    failures = verify_manifest()
    print(json.dumps({"status": "passed" if not failures else "failed", "failures": failures}, indent=2))
    raise SystemExit(0 if not failures else 1)

