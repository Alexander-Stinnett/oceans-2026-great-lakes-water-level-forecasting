from pathlib import Path

from oceans_glwl.paper import reproduce_paper

if __name__ == "__main__":
    result = reproduce_paper(root=Path(__file__).resolve().parents[1], write_outputs=True)
    print(f"Generated {result['rq1_rows'] + result['rq2_rows'] + result['rq3_rows']} table rows.")

