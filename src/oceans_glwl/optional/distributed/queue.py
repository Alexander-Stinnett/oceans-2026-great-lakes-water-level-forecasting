from __future__ import annotations

import os
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    case,
    func,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from oceans_glwl.conditions import FinalSearchCondition

QUEUE_SCHEMA_VERSION = 2
CONDITION_STATUSES = {"pending", "running", "satisfied", "failed"}


class FencingViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class ConditionLease:
    condition_id: str
    study_name: str
    scientific_config: dict[str, Any]
    target_n_trials: int
    worker_id: str
    fencing_token: int
    lease_expires_at: Any
    claim_started_at: Any


class ConditionQueue:
    def __init__(self, engine: Engine, *, schema: str = "oceans_final_search"):
        self.engine = engine
        self.schema = schema
        self.metadata = MetaData(schema=schema)
        self.conditions = Table(
            "conditions",
            self.metadata,
            Column("condition_id", String(255), primary_key=True),
            Column("study_name", String(512), nullable=False, unique=True),
            Column("scientific_config", JSON, nullable=False),
            Column("status", String(32), nullable=False),
            Column("target_n_trials", Integer, nullable=False),
            Column("worker_id", String(128)),
            Column("lease_expires_at", DateTime(timezone=True)),
            Column("fencing_token", BigInteger, nullable=False, default=0),
            Column("git_commit", String(64)),
            Column("artifact_host", String(255)),
            Column("artifact_path", Text),
            Column("last_error", Text),
            Column("status_metadata", JSON),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("started_at", DateTime(timezone=True)),
            Column("claim_started_at", DateTime(timezone=True)),
            Column("completed_at", DateTime(timezone=True)),
            Column("condition_wall_seconds", Float),
        )
        self.workers = Table(
            "workers",
            self.metadata,
            Column("worker_id", String(128), primary_key=True),
            Column("machine_name", String(32), nullable=False),
            Column("backend", String(16), nullable=False),
            Column("hostname", String(255), nullable=False),
            Column("pid", Integer, nullable=False),
            Column("git_commit", String(64), nullable=False),
            Column("status", String(32), nullable=False),
            Column("started_at", DateTime(timezone=True), nullable=False),
            Column("last_seen_at", DateTime(timezone=True), nullable=False),
        )
        self.schema_versions = Table(
            "schema_versions",
            self.metadata,
            Column("component", String(64), primary_key=True),
            Column("version", Integer, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

    def initialize_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:lock_name))"),
                {"lock_name": f"oceans_final_search_schema::{self.schema}"},
            )
            self.metadata.create_all(connection)
            connection.execute(
                pg_insert(self.schema_versions)
                .values(component="queue", version=QUEUE_SCHEMA_VERSION, updated_at=func.now())
                .on_conflict_do_nothing(index_elements=[self.schema_versions.c.component])
            )
            stored = connection.execute(
                select(self.schema_versions.c.version).where(self.schema_versions.c.component == "queue")
            ).scalar_one()
            if int(stored) == 1:
                connection.execute(
                    text(
                        f'ALTER TABLE "{self.schema}"."conditions" '
                        "ADD COLUMN IF NOT EXISTS claim_started_at TIMESTAMP WITH TIME ZONE"
                    )
                )
                connection.execute(
                    text(
                        f'ALTER TABLE "{self.schema}"."conditions" '
                        "ADD COLUMN IF NOT EXISTS condition_wall_seconds DOUBLE PRECISION"
                    )
                )
                connection.execute(
                    update(self.schema_versions)
                    .where(self.schema_versions.c.component == "queue")
                    .values(version=QUEUE_SCHEMA_VERSION, updated_at=func.now())
                )
                stored = QUEUE_SCHEMA_VERSION
            if int(stored) != QUEUE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Queue schema version {stored} is incompatible with code version {QUEUE_SCHEMA_VERSION}; "
                    "apply an explicit migration before continuing."
                )

    def insert_conditions(self, conditions: Iterable[FinalSearchCondition], *, target_n_trials: int) -> int:
        items = list(conditions)
        rows = [
            {
                "condition_id": item.condition_id,
                "study_name": item.study_name,
                "scientific_config": item.to_dict(),
                "status": "pending",
                "target_n_trials": int(target_n_trials),
                "fencing_token": 0,
                "status_metadata": {},
                "created_at": func.now(),
                "updated_at": func.now(),
            }
            for item in items
        ]
        statement = (
            pg_insert(self.conditions)
            .values(rows)
            .on_conflict_do_nothing(index_elements=[self.conditions.c.condition_id])
            .returning(self.conditions.c.condition_id)
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
            inserted_ids = list(result.scalars())
            existing = connection.execute(
                select(self.conditions.c.condition_id, self.conditions.c.scientific_config).where(
                    self.conditions.c.condition_id.in_([item.condition_id for item in items])
                )
            ).mappings()
            stored = {row["condition_id"]: row["scientific_config"] for row in existing}
        for item in items:
            if stored.get(item.condition_id) != item.to_dict():
                raise RuntimeError(f"Stored scientific configuration drift for {item.condition_id}.")
        return len(inserted_ids)

    def register_worker(self, *, worker_id: str, machine_name: str, backend: str, git_commit: str) -> None:
        statement = pg_insert(self.workers).values(
            worker_id=worker_id,
            machine_name=machine_name,
            backend=backend,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            git_commit=git_commit,
            status="idle",
            started_at=func.now(),
            last_seen_at=func.now(),
        ).on_conflict_do_update(
            index_elements=[self.workers.c.worker_id],
            set_={
                "machine_name": machine_name,
                "backend": backend,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "git_commit": git_commit,
                "status": "idle",
                "started_at": func.now(),
                "last_seen_at": func.now(),
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)

    def claim_next(self, *, worker_id: str, git_commit: str, lease_seconds: int) -> ConditionLease | None:
        if int(lease_seconds) <= 0:
            raise ValueError("lease_seconds must be positive.")
        with self.engine.begin() as connection:
            candidate = connection.execute(
                select(self.conditions)
                .where(
                    or_(
                        self.conditions.c.status == "pending",
                        and_(
                            self.conditions.c.status == "running",
                            self.conditions.c.lease_expires_at < func.now(),
                        ),
                    )
                )
                .order_by(self.conditions.c.updated_at, self.conditions.c.condition_id)
                .with_for_update(skip_locked=True)
                .limit(1)
            ).mappings().first()
            if candidate is None:
                return None
            token = int(candidate["fencing_token"]) + 1
            lease_expires = func.now() + timedelta(seconds=int(lease_seconds))
            connection.execute(
                update(self.conditions)
                .where(self.conditions.c.condition_id == candidate["condition_id"])
                .values(
                    status="running",
                    worker_id=worker_id,
                    fencing_token=token,
                    lease_expires_at=lease_expires,
                    git_commit=git_commit,
                    last_error=None,
                    started_at=func.coalesce(self.conditions.c.started_at, func.now()),
                    claim_started_at=func.now(),
                    completed_at=None,
                    condition_wall_seconds=None,
                    updated_at=func.now(),
                )
            )
            connection.execute(
                update(self.workers).where(self.workers.c.worker_id == worker_id).values(
                    status="running", last_seen_at=func.now()
                )
            )
            claimed = connection.execute(
                select(self.conditions).where(self.conditions.c.condition_id == candidate["condition_id"])
            ).mappings().one()
        return ConditionLease(
            condition_id=str(claimed["condition_id"]),
            study_name=str(claimed["study_name"]),
            scientific_config=dict(claimed["scientific_config"]),
            target_n_trials=int(claimed["target_n_trials"]),
            worker_id=worker_id,
            fencing_token=int(claimed["fencing_token"]),
            lease_expires_at=claimed["lease_expires_at"],
            claim_started_at=claimed["claim_started_at"],
        )

    def assert_ownership(self, lease: ConditionLease) -> None:
        with self.engine.connect() as connection:
            valid = connection.execute(
                select(self.conditions.c.condition_id).where(self._ownership_predicate(lease, require_unexpired=True))
            ).first()
        if valid is None:
            raise FencingViolation(f"Lease ownership lost for {lease.condition_id} token={lease.fencing_token}.")

    def current_target(self, lease: ConditionLease) -> int:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.conditions.c.target_n_trials).where(
                    self._ownership_predicate(lease, require_unexpired=True)
                )
            ).first()
        if row is None:
            raise FencingViolation(f"Lease ownership lost for {lease.condition_id} token={lease.fencing_token}.")
        return int(row[0])

    def heartbeat(self, lease: ConditionLease, *, lease_seconds: int) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.conditions)
                .where(self._ownership_predicate(lease, require_unexpired=True))
                .values(lease_expires_at=func.now() + timedelta(seconds=int(lease_seconds)), updated_at=func.now())
            )
            self._require_guarded_update(result.rowcount, lease, "heartbeat")
            connection.execute(
                update(self.workers).where(self.workers.c.worker_id == lease.worker_id).values(last_seen_at=func.now())
            )

    def finalize(
        self,
        lease: ConditionLease,
        *,
        artifact_host: str,
        artifact_path: str,
        complete_trials: int,
        condition_wall_seconds: float,
    ) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.conditions)
                .where(
                    self._ownership_predicate(lease, require_unexpired=True),
                    self.conditions.c.target_n_trials <= int(complete_trials),
                )
                .values(
                    status="satisfied",
                    artifact_host=artifact_host,
                    artifact_path=artifact_path,
                    status_metadata={
                        "complete_trials": int(complete_trials),
                        "monotonic_wall_seconds": float(condition_wall_seconds),
                    },
                    worker_id=None,
                    lease_expires_at=None,
                    completed_at=func.now(),
                    condition_wall_seconds=func.extract(
                        "epoch", func.now() - self.conditions.c.claim_started_at
                    ),
                    updated_at=func.now(),
                )
            )
            self._require_guarded_update(result.rowcount, lease, "finalize")
            connection.execute(
                update(self.workers).where(self.workers.c.worker_id == lease.worker_id).values(
                    status="idle", last_seen_at=func.now()
                )
            )

    def mark_failed(self, lease: ConditionLease, *, error: str, condition_wall_seconds: float | None = None) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.conditions)
                .where(self._ownership_predicate(lease, require_unexpired=True))
                .values(
                    status="failed",
                    last_error=str(error)[:8000],
                    worker_id=None,
                    lease_expires_at=None,
                    completed_at=func.now(),
                    condition_wall_seconds=func.extract(
                        "epoch", func.now() - self.conditions.c.claim_started_at
                    ),
                    status_metadata={"monotonic_wall_seconds": condition_wall_seconds},
                    updated_at=func.now(),
                )
            )
        self._require_guarded_update(result.rowcount, lease, "mark_failed")

    def release(self, lease: ConditionLease) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.conditions)
                .where(self._ownership_predicate(lease, require_unexpired=True))
                .values(status="pending", worker_id=None, lease_expires_at=None, updated_at=func.now())
            )
        self._require_guarded_update(result.rowcount, lease, "release")

    def reclaim_stale(self) -> int:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.conditions)
                .where(
                    self.conditions.c.status == "running",
                    self.conditions.c.lease_expires_at < func.now(),
                )
                .values(
                    status="pending",
                    worker_id=None,
                    lease_expires_at=None,
                    fencing_token=self.conditions.c.fencing_token + 1,
                    last_error="lease expired and was reclaimed",
                    updated_at=func.now(),
                )
            )
        return int(result.rowcount or 0)

    def requeue_failed(self, *, condition_id: str) -> int:
        statement = (
            update(self.conditions)
            .where(
                self.conditions.c.condition_id == condition_id,
                self.conditions.c.status == "failed",
            )
            .values(
                status="pending",
                worker_id=None,
                lease_expires_at=None,
                fencing_token=self.conditions.c.fencing_token + 1,
                git_commit=None,
                artifact_host=None,
                artifact_path=None,
                last_error=None,
                status_metadata={},
                claim_started_at=None,
                completed_at=None,
                condition_wall_seconds=None,
                updated_at=func.now(),
            )
            .returning(self.conditions.c.condition_id)
        )
        with self.engine.begin() as connection:
            requeued_ids = list(connection.execute(statement).scalars())
        return len(requeued_ids)

    def requeue_all_failed(self) -> int:
        statement = (
            update(self.conditions)
            .where(self.conditions.c.status == "failed")
            .values(
                status="pending",
                worker_id=None,
                lease_expires_at=None,
                fencing_token=self.conditions.c.fencing_token + 1,
                git_commit=None,
                last_error=None,
                status_metadata={},
                claim_started_at=None,
                completed_at=None,
                condition_wall_seconds=None,
                updated_at=func.now(),
            )
            .returning(self.conditions.c.condition_id)
        )
        with self.engine.begin() as connection:
            requeued_ids = list(connection.execute(statement).scalars())
        return len(requeued_ids)

    def set_target(self, *, target_n_trials: int, condition_id: str | None = None) -> int:
        if int(target_n_trials) <= 0:
            raise ValueError("target_n_trials must be positive.")
        predicate = self.conditions.c.target_n_trials != int(target_n_trials)
        if condition_id is not None:
            predicate = and_(predicate, self.conditions.c.condition_id == condition_id)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.conditions)
                .where(predicate)
                .values(
                    target_n_trials=int(target_n_trials),
                    status=case(
                        (
                            and_(
                                self.conditions.c.status == "satisfied",
                                self.conditions.c.target_n_trials < int(target_n_trials),
                            ),
                            "pending",
                        ),
                        else_=self.conditions.c.status,
                    ),
                    completed_at=None,
                    updated_at=func.now(),
                )
            )
        return int(result.rowcount or 0)

    def rows(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(self.conditions)).mappings()]

    def get_condition(self, *, condition_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.conditions).where(self.conditions.c.condition_id == condition_id)
            ).mappings().one_or_none()
        if row is None:
            raise KeyError(f"Unknown OCEANS final-search condition: {condition_id}.")
        return dict(row)

    def workers_rows(self) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(self.workers)).mappings()]

    def _ownership_predicate(self, lease: ConditionLease, *, require_unexpired: bool) -> Any:
        predicate = and_(
            self.conditions.c.condition_id == lease.condition_id,
            self.conditions.c.status == "running",
            self.conditions.c.worker_id == lease.worker_id,
            self.conditions.c.fencing_token == lease.fencing_token,
        )
        if require_unexpired:
            predicate = and_(predicate, self.conditions.c.lease_expires_at >= func.now())
        return predicate

    @staticmethod
    def _require_guarded_update(rowcount: int | None, lease: ConditionLease, operation: str) -> None:
        if int(rowcount or 0) != 1:
            raise FencingViolation(
                f"Fencing violation during {operation} for {lease.condition_id} token={lease.fencing_token}."
            )
