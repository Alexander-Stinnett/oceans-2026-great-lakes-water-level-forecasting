import sys
from pathlib import Path

from oceans_glwl.paper import reproduce_paper


def test_check_only_paper_reproduction_has_no_training_imports(repo_root: Path) -> None:
    for name in ("torch", "neuralforecast", "optuna", "sqlalchemy", "psycopg"):
        sys.modules.pop(name, None)
    result = reproduce_paper(root=repo_root, write_outputs=False)
    assert result["status"] == "success"
    assert result["golden_checks"] == {"rq1_values": 8, "rq2_values": 19, "rq3_printed_values": 90}
    assert not any(name in sys.modules for name in ("torch", "neuralforecast", "optuna", "sqlalchemy", "psycopg"))

