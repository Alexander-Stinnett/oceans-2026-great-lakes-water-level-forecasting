# Historical execution

The final search used Optuna studies backed by PostgreSQL plus a separate
condition queue with leases, heartbeats, and fencing tokens. Sky and Lavvy used
CUDA; Mac used MPS. Those names and accelerators are provenance, not current
execution requirements.

Optional modules under `oceans_glwl.optional.distributed` preserve database
settings, queue semantics, idempotent planning, host export/aggregation, and a
worker adapter. Install the `distributed` extra to import them. They are not
imported by data processing, analysis, paper reproduction, or local training.

The optional worker deliberately leaves final artifact publication to the
deployment wrapper so a lease is not marked satisfied before verified outputs
exist.

