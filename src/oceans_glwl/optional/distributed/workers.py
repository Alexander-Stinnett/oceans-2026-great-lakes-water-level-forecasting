from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oceans_glwl.conditions import condition_from_dict
from oceans_glwl.optional.distributed.queue import ConditionQueue
from oceans_glwl.training.hpo import run_local_hpo


@dataclass(frozen=True)
class DistributedWorkerConfig:
    worker_id: str
    data_path: Path
    results_root: Path
    device: str = "auto"
    lease_seconds: int = 600
    max_steps: int = 750
    git_commit: str = "unknown"


def run_one_claim(
    queue: ConditionQueue,
    *,
    optuna_storage_url: str,
    config: DistributedWorkerConfig,
) -> dict[str, Any] | None:
    """Claim one historical condition and extend its persistent Optuna study.

    This optional adapter preserves the PostgreSQL lease/fencing ownership
    model. Artifact publication remains deployment-specific; callers must
    finalize the lease only after publishing a complete attempt directory.
    """
    lease = queue.claim_next(
        worker_id=config.worker_id,
        git_commit=config.git_commit,
        lease_seconds=config.lease_seconds,
    )
    if lease is None:
        return None
    condition = condition_from_dict(lease.scientific_config)
    try:
        study = run_local_hpo(
            config.data_path,
            condition=condition,
            n_trials=lease.target_n_trials,
            device=config.device,
            max_steps=config.max_steps,
            storage=optuna_storage_url,
        )
    except BaseException as exc:
        queue.mark_failed(lease, error=repr(exc))
        raise
    return {
        "lease": lease,
        "condition_id": condition.condition_id,
        "complete_trials": len([trial for trial in study.trials if trial.state.name == "COMPLETE"]),
        "study": study,
    }

