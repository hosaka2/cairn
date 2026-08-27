# Usage

cairn is a **ledger for evaluations and datasets**. With no DB to stand up, it uses object storage
(S3 / GCS / local) alone to record and compare append-only datasets and the evaluation results that stack up on them.

Training and model management (MLflow / W&B), LLM tracing (Langfuse), and running inference (Dagster etc.) are out of scope.

## 1. Decide where it lives

Everything lives under `CAIRN_ROOT`. Resolution order: **`--root` > environment variable > `.env` in the current directory**.

```bash
cairn --root file:///tmp/cairn-demo dataset ls     # flag
export CAIRN_ROOT=s3://bucket/cairn                 # environment variable
```
```env
# .env (read automatically when placed in the current directory)
CAIRN_ROOT=s3://bucket/cairn
AWS_PROFILE=myprofile        # credentials are picked up automatically from env vars / ~/.aws / IAM roles
AWS_ENDPOINT_URL=http://localhost:9000   # only when using MinIO
CAIRN_LANG=ja                            # manual and form help in Japanese (English unless set)
CAIRN_TRACEBACK=1                        # show the traceback instead of the message (debugging a script)
```

## 2. Get it running

```bash
cairn demo-seed     # run every feature for real on synthetic data (nothing faked)
cairn web           # http://127.0.0.1:8000
```

## 3. The three actors

| | Role | Unit |
|---|---|---|
| **Dataset** | What gets evaluated. A ledger you **append to freely** to build it up | 1 row = 1 sample |
| **Eval table** | The definition of what is measured and by which metrics. Decides the columns of the list | 1 table = 1 kind of evaluation |
| **Run** | One measurement. Writes predictions, evaluates, and leaves the results behind | 1 run = 1 stone |

In the eval list, **runs stack up as stones** (stone width = primary metric).

## 4. One lap through the screens

1. **Create a dataset** — paste `schema.yaml` (the column definitions). `key` is the identity of a row.
2. **Add data** — the ingest script form, or pasted JSONL.
   Check the preview → nothing is appended until you confirm. Re-appending the same `key` overwrites (last write wins).
3. **Create an eval table** — paste `table.yaml` (the columns shown in the list). The target dataset is selectable.
4. **New run** — set the title (required), dataset, evaluation method, and config, and issue it.
   cairn does not perform the inference: the run waits under **Running** until predictions are
   written against it (see 5). With the demo scripts, `cairn eval run <table> <eval_id>` writes them.
5. **Evaluate** — press it on the waiting run once the predictions are in. The result is stacked as a stone.
6. **Compare** — select a run and choose "compare with another run". Rows are added to the result table and
   differences show up colored (blue = better direction / red = worse direction; the direction is each column's `direction`).

## 5. Driving a run from an orchestrator

cairn never performs the inference. It happens wherever you run it — an orchestrator (Dagster,
Airflow, a batch queue), a script of your own — and cairn holds the two ends of it:

```bash
# 1. Start the run: this pins the data. "Run" in the web form does the same.
EID=$(cairn eval create-run anomaly --dataset sensor-anomaly-A --evaluator-version v1 \
        --title "nightly batch" --config '{"weights": "s3://models/2026-08-26.pt"}')

# 2. Somewhere else, per task: what to work on, and where the answers go.
cairn eval targets anomaly $EID                  # sample ids, one per line
cairn eval put-prediction anomaly $EID --sample-id A_007 --file pred.json --ext json

# 3. When every task is done, score once.
cairn eval score anomaly $EID --evaluator evals.anomaly.v1:EVALUATOR
```

The run appears under **Running** on the eval page from step 1, with an **Evaluate** button once an
evaluator for its version is registered. How far the jobs have got is the orchestrator's business, not
the ledger's; cairn only knows the run started and has no result yet. The snapshot is pinned at step
1, so appending to the dataset while the jobs run does not change what this run is measured on.

Predictions are opaque bytes: only the evaluator of that table reads them. `--jsonl` writes a batch of
`{"sample_id": …, "prediction": …}` lines in one call. Both forms refuse a run that is already
evaluated: its result stands on the predictions that were there when it was written.

## 6. When results are comparable

cairn engraves results along **two axes**.

- **`snapshot_id`** — the dataset contents at run time (a hash of the contents). **Only runs that match can be compared.**
  Re-ingesting the same rows does not change it, but fixing a value or deleting a row does.
  The UI still lets you put a run on other data side by side — sometimes you want to see it — and marks that row `other data`.
- **`evaluator_version`** — the version of the evaluation formula (`v1`, `v2`, …). Always cut a new version when you change the formula.
  This is what tells you whether "0.81 last month / 0.79 this month" is a change in the data or a change in the formula.

Model identity (weight version, hyperparameters) has no dedicated field; put it in the **run's `config`**, where it is
stored together with `code_commit`.

## 7. CLI

```bash
cairn check                     # check that scripts follow the contract (run this first after writing one)
cairn init                      # create datasets/ and evals/
cairn new dataset <name>        # scaffold schema.yaml + ingest.py
cairn new eval <name>           # scaffold table.yaml + run.py + v1.py
cairn new eval-version <name> v2  # new version of the evaluation formula (never edit the old one)

cairn dataset ls
cairn dataset create --schema schema.yaml
cairn dataset ingest <name> --jsonl rows.jsonl   # use this for bulk data (the web paste box is for small amounts)
cairn dataset show <name>
cairn dataset delete <name> --key a --key b        # tombstones; changes the snapshot

cairn eval ls
cairn eval create-table --table table.yaml
cairn eval create-run <table> --dataset <ds> --evaluator-version v1 \
    --title "what this run is" --config '{"threshold": 0.5}'
cairn eval targets <table> <eval_id>               # the sample ids this run is pinned to
cairn eval put-prediction <table> <eval_id> --sample-id a --file pred.json --ext json
cairn eval put-prediction <table> <eval_id> --jsonl preds.jsonl   # batch: one prediction per line
cairn eval run <table> <eval_id>                   # perform it here with the registered script
cairn eval score <table> <eval_id> --evaluator my.module:MyEvaluator
cairn eval withdraw <table> <eval_id> --reason "wrong weights"
                                # take a run out of the listings (final; the record stays)
cairn eval rescore <table> <eval_id> --evaluator my.module:V2 --evaluator-version v2
                                # score the same predictions another way, as a new run
cairn eval show <table> <eval_id>

cairn vacuum                    # reclaim old checkpoints (derived garbage; no data is deleted)
cairn docs scripting            # how to write scripts
```

## 8. Where to run the web app

It binds `127.0.0.1` and has no login: anyone who reaches it can add data, start runs and
evaluate them. Keep it on localhost, or put it behind a proxy that authenticates — that is
where `auth_header` (`X-Forwarded-User`, the name recorded as `created_by`) comes from.

## 9. How it works (knowing this keeps you out of trouble)

- Writes are always stacked as **a new file** and never overwrite. That is why no locking is needed.
- A dataset's current state is the result of applying the append log in order (same `key` is last write wins, deletes are tombstones).
- A dataset's rows are listed through a parquet checkpoint, but **the truth is on the JSON side**; it can be rebuilt at any
  time. A table's results are read from the JSON itself, one file per run.
- `cairn vacuum` deletes only the cache (old checkpoints). Data referenced by runs is never deleted.
