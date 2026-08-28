import argparse
from pathlib import Path

from oceans_glwl.data.build import rebuild_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("data/upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(rebuild_dataset(args.upstream, args.output, overwrite=args.overwrite))

