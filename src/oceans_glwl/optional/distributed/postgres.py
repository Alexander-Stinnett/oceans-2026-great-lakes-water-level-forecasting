from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

import optuna
from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback
from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.engine import make_url

DEFAULT_DATABASE = "glm_optuna"
DEFAULT_USER = "horizon"
DEFAULT_SCHEMA = "oceans_final_search"
MACHINE_PORTS = {"sky": 5432, "lavvy": 15432, "mac": 15432}


@dataclass(frozen=True)
class DatabaseSettings:
    url: URL
    schema: str = DEFAULT_SCHEMA
    heartbeat_interval: int = 60
    grace_period: int = 180
    stale_trial_retries: int = 3

    @classmethod
    def from_environment(
        cls,
        *,
        machine_name: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> DatabaseSettings:
        env = os.environ if environ is None else environ
        explicit = env.get("HORIZON_DATABASE_URL")
        if explicit:
            url = make_url(explicit)
        else:
            password = env.get("HORIZON_DB_PASSWORD")
            if not password:
                raise RuntimeError("Set HORIZON_DATABASE_URL or HORIZON_DB_PASSWORD.")
            machine = str(machine_name or env.get("HORIZON_MACHINE") or "").strip().lower()
            configured_port = env.get("HORIZON_DB_PORT")
            if configured_port is None and machine not in MACHINE_PORTS:
                raise RuntimeError("Set HORIZON_MACHINE to sky, lavvy, or mac when constructing the database URL.")
            url = URL.create(
                "postgresql+psycopg",
                username=env.get("HORIZON_DB_USER", DEFAULT_USER),
                password=password,
                host=env.get("HORIZON_DB_HOST", "127.0.0.1"),
                port=int(configured_port or MACHINE_PORTS[machine]),
                database=env.get("HORIZON_DB_NAME", DEFAULT_DATABASE),
            )
        schema = env.get("HORIZON_DB_SCHEMA", DEFAULT_SCHEMA)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema) is None:
            raise ValueError(f"Invalid PostgreSQL schema name: {schema!r}.")
        heartbeat = int(env.get("HORIZON_OPTUNA_HEARTBEAT_SECONDS", "60"))
        grace = int(env.get("HORIZON_OPTUNA_GRACE_SECONDS", "180"))
        retries = int(env.get("HORIZON_OPTUNA_STALE_RETRIES", "3"))
        if heartbeat <= 0 or grace <= heartbeat or retries < 0:
            raise ValueError("Optuna heartbeat/grace/retry settings are invalid.")
        return cls(url=url, schema=schema, heartbeat_interval=heartbeat, grace_period=grace, stale_trial_retries=retries)

    @property
    def redacted_url(self) -> str:
        return self.url.render_as_string(hide_password=True)


def create_queue_engine(settings: DatabaseSettings) -> Engine:
    return create_engine(settings.url, pool_pre_ping=True, future=True)


def create_optuna_storage(settings: DatabaseSettings) -> RDBStorage:
    return RDBStorage(
        url=settings.url.render_as_string(hide_password=False),
        engine_kwargs={"pool_pre_ping": True},
        heartbeat_interval=settings.heartbeat_interval,
        grace_period=settings.grace_period,
        heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(
            max_retry=settings.stale_trial_retries,
            inherit_intermediate_values=False,
        ),
    )


def complete_trial_count(study: optuna.Study) -> int:
    return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
