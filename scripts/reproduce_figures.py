from pathlib import Path

from oceans_glwl.artifacts.io import final_search_root
from oceans_glwl.plotting.rq1_predictions import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(run(final_search_root(root), root / "paper/figures/generated"))

