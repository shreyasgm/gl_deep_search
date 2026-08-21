# GL Deep Search — Status as of 2026-08-21

Written after a ~6 month gap. Last commit was **2026-02-25**; last cluster job was **2026-02-25 21:08 EST**; last local activity was **2026-03-11**.

*This replaces the November 2025 status doc, which had gone badly stale — it still described the OpenAI-embeddings era, claimed the service layer and frontend did not exist, and led with a chunker token-limit bug that has since been fixed.*

---

## TL;DR

The project did not stall on a hard problem. It stalled **two minutes into a re-run, on a stale container image**, and then got dropped.

Three things are true right now:

1. **The ETL pipeline works and has been proven at production scale.** A full run finished on the cluster on Feb 25 — 449 publications scraped, 417 files downloaded, 331 PDFs extracted with Marker on an A100, 24,039 chunks, 19,765 Qwen3 embeddings. 17.2 hours wall clock.
2. **That run's output is gone.** Only the raw PDFs (1.1 GB, 414 publications) and the tracking database survive on the cluster. `data/processed/` — the extracted text, chunks, and embeddings — is not there. Re-running costs ~13 GPU-hours of PDF extraction plus ~4 hours of embedding.
3. **Nothing has ever reached Qdrant.** The tracking DB shows `ingestion_status = PENDING` for all 449 rows. There is no Qdrant instance running locally, in Docker Compose, or in GCP. The search API, the LangGraph agent, and the Streamlit frontend all exist and are unit-tested, but have never been run against real data.

So the system still cannot answer a search query — the same headline as November — but the gap is now one integration step wide, not three components wide.

**Update, end of 2026-08-21:** points 1 and 2 are being resolved right now — a full-corpus run (job `40971461`, ~24h) is in flight after a staged rollout that fixed six distinct blockers. Point 3 is untouched and is the real remaining work: **stand up Qdrant and ingest.** See [Staged rollout](#staged-rollout-2026-08-21) and [What to do next](#what-to-do-next).

---

## How it stopped

All times EST, all on 2026-02-25 unless noted.

| Time | Event |
|---|---|
| Feb 24 ~21:39 | Job `62012506` starts — full production ETL run |
| 13:26 | `aff6176` — GPU memory cleanup between pipeline stages |
| **14:54** | **Job `62012506` finishes successfully.** 17h 14m. Report written, results synced back to `holystore01` |
| 16:34 | `d5651cf` — OpenAlex file downloader repaired (scidownl API, Unpaywall email) |
| 17:55 | `0944528` — OpenAlex + lectures wired into the orchestrator; **`--sources` flag added** |
| 20:41 | `9a7de2e` — `etl-lite` dep group; **sbatch updated to pass `--sources ${SOURCES}`** |
| 20:51 | 24 lecture transcripts copied to the cluster |
| 20:53 | `git pull` on the cluster — code now at `9a7de2e` |
| 20:54 | Job `62305984` submitted, cancelled 22 seconds in during data staging |
| 20:59 | New `.sif` pulled from Artifact Registry |
| 21:06 | Job `62310653` submitted |
| **21:08** | **Dies in 2 minutes:** `orchestrator.py: error: unrecognized arguments: --sources all` |
| Mar 2 | `.github/workflows/python-checks.yml` edited locally — never committed |
| Mar 10–11 | Local dev-mode runs (OpenAlex + lectures, small scale) |
| — | Silence |

### Root cause of the final failure

Not a code bug — a **build/deploy skew**. The `.sif` on the cluster was pulled at 20:59, but the image behind the `latest` tag in Artifact Registry had been built *before* commit `0944528`, which is the commit that added `--sources`. Verified directly:

```
$ singularity exec gl-pdf-processing.sif grep -c -- "--sources" /app/backend/etl/orchestrator.py
0
$ singularity exec gl-pdf-processing.sif ls -la /app/backend/etl/orchestrator.py
-rw-r--r-- 1 root root 36358 Feb 25 17:51 /app/backend/etl/orchestrator.py

$ wc -c backend/etl/orchestrator.py      # local, current main
46268
```

The baked-in orchestrator is 36 KB; current `main` is 46 KB. The sbatch script (from `9a7de2e`) passes a flag the container's Python has never heard of.

**Fix: rebuild the image, re-pull the `.sif`.** Two commands, ~20 minutes of Cloud Build. That is the entire blocker.

---

## Current state by layer

| Layer | State | Notes |
|---|---|---|
| **Growth Lab scraper** | ✅ Working | 449 publications, 452 file URLs |
| **GL file downloader** | ✅ Working | 417/452 succeeded (92%) |
| **OpenAlex scraper + downloader** | ✅ Ran on cluster 2026-08-21 | 332 publications scraped in the smoke test. Note the scraper ignores `--scraper-limit`; only the downloader honours a limit |
| **PDF processor (Marker/CUDA)** | ✅ Working | 331/336 extracted. 13.1 hours on an A100 — the dominant cost |
| **Lecture transcripts** | ✅ Ran on cluster 2026-08-21 | 1/1 in the smoke test; all 24 included in the production run. Each makes an LLM call — cap with `TRANSCRIPTS_LIMIT` |
| **Text chunker** | ✅ Working | 24,039 chunks, 0 failures. The Nov token-limit bug is fixed |
| **Embeddings (Qwen3-8B local)** | ✅ Fixed 2026-08-21 | Feb: 266/304, all 38 failures CUDA OOM. Now loads in bf16 with a capped sequence length; work list is disk-driven so nothing is silently dropped |
| **Qdrant ingestion** | 🔴 Never executed | `ingest_to_qdrant.py` exists and is unit-tested. `ingestion_status = PENDING` × 449 |
| **Vector DB instance** | 🔴 Does not exist | No local container, no Compose service, no cloud instance. `qdrant_url` defaults to `localhost:6333` |
| **Search API (FastAPI)** | 🟡 Written, never run live | 3 endpoints, tested with `TestClient` and mocks |
| **LangGraph agent** | 🟡 Written, never run live | analyze → retrieve → grade → synthesize; tests added Feb 24 |
| **Streamlit frontend** | 🟡 Written, never run live | `frontend/app.py` + `api_client.py` |
| **SLURM deployment** | ✅ Unblocked | Image rebuilt and verified; commit-stamped with a path-scoped staleness guard. 48h limit, `PDF_LIMIT`/`TRANSCRIPTS_LIMIT` for bounded test runs |
| **GCP / Cloud Run** | 🟡 Idle | Job `etl-pipeline-job` exists, last run 2025-11-14. Not scheduled, not costing anything meaningful |
| **Tests** | ✅ Healthy | **282 passed, 3 skipped** (integration deselected; was 279 before today's additions). `ruff check`, `ruff format --check`, `mypy` all clean — but only once the `service` extra is installed |

### Repo hygiene

- `main` is clean and pushed except one uncommitted file: `.github/workflows/python-checks.yml`, which adds `--extra service` to the CI dependency install. **This change is correct and necessary** — without it CI cannot even import the service tests (`fastembed`, `starlette` missing). It has been sitting uncommitted since Mar 2.
- 15 stale remote branches (`fastapi-endpoint`, `embeddings`, `ocr_pipe`, `manifest-branch`, `feat/chunk-tagger`, …), most predating the February work.
- Five audit documents (`audit_*.md`, `test_fixes_summary.md`, `test_strategy_rework_review.md`) are sitting at the repo root. They're valuable but belong in `docs/`.

---

## Data inventory

### Cluster — `/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search/`

| Item | State |
|---|---|
| `data/raw/documents/growthlab/` | **1.1 GB, 414 publication dirs** — intact |
| `data/raw/lecture_transcripts/` | 24 `.txt` files — intact, never processed |
| `data/etl_tracking.db` | 1.4 MB, 449 rows — intact |
| `data/processed/` | **Missing** |
| `reports/` | Empty |
| `deployment/slurm/gl-pdf-processing.sif` | 4.6 GB — **stale, needs rebuild** |
| `.model_cache/` | Present — Qwen3 + Marker weights cached, saves a re-download |

**On the missing `processed/` directory:** the Feb 25 run log confirms it was written to `/app/data/processed/...` and that the sync-back to persistent storage completed. The sbatch cleanup uses `rsync -a` *without* `--delete`, so it could not have removed it. Most likely you cleared it manually between 14:55 and 20:51 to force a clean re-run with the new OpenAlex/lectures wiring. Worth a moment's thought before re-running — if it was moved rather than deleted, that's 13 GPU-hours recovered.

### Tracking DB status counts (449 publications)

| Stage | Done | Pending | Failed |
|---|---|---|---|
| Download | 379 | 35 | 35 |
| PDF processing | 331 | 113 | 5 |
| Embedding | 293 | 118 | 38 |
| **Qdrant ingestion** | **0** | **449** | **0** |

Note the DB is now out of sync with disk: it says 331 documents are `PROCESSED` and 293 `EMBEDDED`, but those artifacts no longer exist. Resume logic is **file-existence based**, not DB based, so a re-run will correctly redo the work — but the DB counts will read as stale until then.

### Local — `gl_deep_search/data/`

393 MB raw, ~1 MB processed. Dev-scale only, from the March runs. Includes `data/processed/documents/openalex/` and lecture transcript output — the March local testing.

### GCS — `gs://gl-deep-search-data/`

200 MB total, all from **2026-02-19** — an earlier pipeline vintage (different chunking config). Superseded, not reusable as a recovery source.

---

## Issues found

Ranked by what actually blocks progress. **Issues 1–3 were fixed on 2026-08-21** — see [Work done today](#work-done-2026-08-21) at the end. They're documented here in full because the diagnosis matters more than the diff.

### 1. Embeddings generator OOMs on an 80 GB A100 — 38 documents lost (13%) — ✅ FIXED

Every one of the 38 failures was `torch.cuda.OutOfMemoryError`. From the logs:

> `Tried to allocate 11.72 GiB. GPU 0 has a total capacity of 79.25 GiB of which 1.35 GiB is free. This process has 77.89 GiB memory in use. Of the allocated memory 55.51 GiB is allocated by PyTorch, and 21.87 GiB is reserved by PyTorch but unallocated.`

55 GB of resident allocation for an 8B embedding model is the tell. Four contributing causes, all in `backend/etl/utils/embeddings_generator.py`:

- **Line 147 — the model is loaded without a dtype.** `SentenceTransformer(self.model_name, trust_remote_code=True)`. No `model_kwargs={"torch_dtype": torch.bfloat16}`. Qwen3-Embedding-8B in fp32 is ~32 GB of weights alone; in bf16 it is ~16 GB. This is almost certainly the main cause.
- **Line 355 — the OOM retry can never reach a small enough batch.** `max_oom_retries = 3` halving from `batch_size=32` reaches 16 → 8 → 4, then gives up. It never gets near 1. Needs ~6 retries, or an explicit floor.
- **`max_seq_length` is never set.** With `max_chunk_size: 8000` in `config.yaml`, outlier chunks push 8k-token sequences through an 8B model. `sentence-transformers` sorts by length internally and processes the longest batch *first*, which is why the failures cluster on the first batch (visible in the `.err` progress bars).
- **`release_gpu_memory()` is only called inside the OOM handler**, not after each document. Fragmentation accumulates across 300+ documents.

Concrete fix: load in bf16, cap `max_seq_length` at ~2048, lower `max_chunk_size` to ~2000 tokens (the chunker targets 500 anyway, so 8000 only ever applies to un-splittable outliers), raise the retry floor to batch size 1, and release memory per document.

### 2. Build/deploy skew has no guard — ✅ FIXED

Nothing detects that the container's code is older than the repo's. This burned a submission, and it will do it again. Cheap fix: bake `git rev-parse HEAD` into the image as a label or `/app/GIT_SHA`, and have the sbatch script compare against the cluster checkout and refuse to run on mismatch.

### 3. sbatch time limit is shorter than the run it has to survive — ✅ FIXED

The sbatch requests `--mem=100G` and `--time=12:00:00`, but the successful run took **17h 14m**. It only completed because the job had a longer allocation than the current script grants — as written today, the same run would hit `TIMEOUT` five hours from the end, mid-embedding. Either raise `-t` to `1-00:00` or split PDF extraction and embedding into separate jobs. Check `jobstats 62012506` for the real memory profile before touching `--mem`.

### 4. Outstanding test-quality debt from the February audit

The audits (`audit_summary.md` and the four detail files) were written Feb 24, and `test_fixes_summary.md` records that much of Tier 1 was subsequently addressed — the agent, `main.py`, `retry.py`, and `ingest_to_qdrant.py` all have tests now, and the suite is genuinely green. What appears **not** to have been resolved:

- `_build_url()` in `openalex.py` — `lstrip('A')` strips *all* leading A's, so author ID `AAB123` becomes `B123` (audit item 19). Worth 5 minutes to confirm and fix.
- Duplicate GCS implementations, `storage/cloud.py` vs `storage/gcs.py` — one is likely dead code (item 20).
- Storage-layer tests (item 10) — `StorageFactory` auto-detection silently routing the whole pipeline to the wrong backend is a real risk given the local/cloud split.

### 4b. scidownl (Sci-Hub) fallback cannot work inside the container — needs a decision

Observed live in the production run: **all 250 scidownl attempts failed**, every one with

```
Error using scidownl: (sqlite3.OperationalError) unable to open database file
```

Root cause, traced to `scidownl/db/entities.py:22-24`: the DB path is computed as
`os.path.join(dirname(dirname(__file__)), configs['global_db']['db_name'])` — i.e.
`<site-packages>/scidownl/scidownl.db`, inside its own package directory. That is read-only in a `.sif`,
and the file does not ship with the package, so `create_tables()` cannot create it. `import scidownl`
raises at import time. `--writable-tmpfs` does not help, and bind-mounting just the DB file is
insufficient because SQLite also needs to write journal files into that directory.

**Impact: OpenAlex downloads are limited to open-access papers.** The Unpaywall/OA path works
normally (2/4 in the smoke test were `open_access_downloads`). Non-OA papers silently fail.
This affects only OpenAlex; the Growth Lab corpus (451 files) is unaffected.

**Fixed 2026-08-21** (Sci-Hub use confirmed acceptable by the project owner). The sbatch now resolves the package path with `importlib.util.find_spec` (it cannot `import scidownl` — that is the bug), copies the package to local scratch, and bind-mounts it back over the original path so the directory is writable. Verified on the cluster: `import scidownl` succeeds and `scidownl.db` is created. Degrades to a warning if the package is missing. **The in-flight run 40971461 predates this**, so its OpenAlex coverage is open-access only; a follow-up run picks up the rest incrementally. The original analysis follows. The fix is easy — copy the package to scratch at job start and
bind-mount it back over the original path so the directory is writable — but scidownl fetches from
Sci-Hub, whose legal status is contested. Whether to invest in making that path work is a call for
the project owner, not an incidental infrastructure fix. Options:

Recovery is incremental: resume is file-existence based, so a later run
downloads only the missing PDFs and extracts only those.

### 5. 35 download failures + 5 extraction failures never triaged

8% of the corpus. Nobody has looked at whether these are dead links, paywalls, or a fixable bug in the downloader.

### 6. Two container facts that bear on the rebuild and on Phase 3

- **The container installs only the `etl` extra** (`deployment/pdf-processing/Dockerfile:40` — `uv sync --locked --no-dev --extra etl`). `ingest_to_qdrant.py` imports `fastembed`, which lives in the `service` extra. **The ingestion step cannot run inside the current image.** Either add `--extra service` to that line, or run ingestion outside the container. Decide this before Phase 3 rather than discovering it mid-run.
- **The builder base image is unpinned** — `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, no version tag. Given that the whole February failure was a skew problem, pinning this is cheap insurance. Not urgent, but note that the next rebuild will pull a newer `uv` than the one that produced the current image.

> Minor: running `uv sync` today added an inert `[options]` block to `uv.lock` (a newer-`uv` metadata artifact). **No pinned versions changed** — verified. Keep it or revert it; it makes no functional difference.

---

## What to do next

Ordered so that each step de-risks the next. Steps 1–3 get you back to where you were in February; 4–6 are the actual new ground.

### Phase 0 — Unblock (≈30 min, mostly waiting)

1. ~~**Commit the CI fix.**~~ Still uncommitted, but verified: with `--extra service` the suite goes from 4 collection errors to fully green. Ready to commit alongside today's changes.
2. **Decide the `data/processed/` question.** ⬅️ *Needs your memory — this is the one thing I can't determine.* Was it deleted or moved? If there's any chance it's recoverable, that's 13 GPU-hours saved.
3. **Rebuild and redeploy the container:**
   ```bash
   bash deployment/slurm/setup_env.sh build          # local — Cloud Build, ~20 min
   ssh ody 'cd $PROJECT_DIR && git pull && bash deployment/slurm/setup_env.sh pull'
   ```
4. **Verify before submitting** — this is the check that would have saved February:
   ```bash
   ssh ody 'cd $PROJECT_DIR && singularity exec deployment/slurm/gl-pdf-processing.sif \
     python -m backend.etl.orchestrator --help | grep -- --sources'
   ```

### Phase 1 — Fix the OOM before spending GPU hours — ✅ DONE

5. ~~Apply the four fixes in Issue 1.~~ Done.
6. ~~Add a regression test that asserts the OOM retry reaches batch size 1.~~ Done — three new tests, suite now at 282.
7. **Still to do:** validate on a short job — `SCRAPER_LIMIT=20 DOWNLOAD_LIMIT=20 sbatch deployment/slurm/etl_pipeline.sbatch`. This is exactly the run that died in February, so it doubles as confirmation of both the OOM fix and the new staleness guard.

### Phase 2 — Re-run production (≈18 h wall clock, mostly unattended)

8. Raise the sbatch time limit to `1-00:00` first (Issue 3).
9. Full run with all three sources: `SOURCES=all sbatch deployment/slurm/etl_pipeline.sbatch`. Downloads will be skipped (raw PDFs intact); PDF extraction and embedding rerun. Lectures and OpenAlex process for the first time.
10. Run `jobstats <JOBID>` afterwards and right-size `--mem` and `-c` in the script.

### Phase 3 — Close the loop to search (the actual new work)

11. **Stand up Qdrant.** Add a `qdrant` service to `docker-compose.yml` for local development. Decide the production target — Qdrant Cloud free tier is likely sufficient at 20k vectors × 1024 dims (~80 MB), and avoids running a VM.
12. **Run `ingest_to_qdrant.py` for real.** It has never executed against real data. Expect schema friction on the first attempt — it merges parquet embeddings, chunk JSON, and tracker rows, and the parquet layout changed when embeddings moved to Qwen3.
13. **Wire ingestion into the orchestrator** as a final component so `ingestion_status` stops being permanently `PENDING`.
14. **Run the API against real data** — `uvicorn backend.service.main:app`, hit `/search/chunks`, then the agent, then Streamlit. First real end-to-end query.

### Phase 4 — Cleanup, when convenient

15. ~~Delete the stale `PROJECT_STATUS.md`.~~ Done — this file replaced it.
16. Move `audit_*.md`, `test_fixes_summary.md`, `test_strategy_rework_review.md` into `docs/`.
17. Prune the 15 stale remote branches.
18. Triage the 35 download and 5 extraction failures.
19. Fix the `lstrip('A')` bug in `openalex.py`.

---

## Runbook

```bash
# ── Local ────────────────────────────────────────────────────────────
uv sync --extra etl --extra service --extra dev   # NOTE: service extra is required
uv run pytest -m "not integration"                # 279 passed, 3 skipped, ~2 min
uv run ruff check . && uv run ruff format --check . && uv run mypy .

uv run python -m backend.etl.orchestrator --dev --sources all --download-limit 3

# ── Container ────────────────────────────────────────────────────────
bash deployment/slurm/setup_env.sh build          # local: Cloud Build → Artifact Registry
ssh ody 'cd /n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search \
         && git pull && bash deployment/slurm/setup_env.sh pull'

# ── Cluster ──────────────────────────────────────────────────────────
ssh ody
cd /n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search

SCRAPER_LIMIT=20 DOWNLOAD_LIMIT=20 sbatch deployment/slurm/etl_pipeline.sbatch   # smoke
SOURCES=all sbatch deployment/slurm/etl_pipeline.sbatch                          # full

squeue --me
tail -f logs/etl_pipeline_<JOBID>.out
jobstats <JOBID>
```

### Cluster reference

| | |
|---|---|
| Host | `ody` → `login.rc.fas.harvard.edu` |
| Project dir | `/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search/` |
| Partition | `gpu` (A100 80 GB), `--gres=gpu:1` |
| Image | `us-east4-docker.pkg.dev/cid-hks-1537286359734/etl-pipeline/gl-pdf-processing:latest` |
| GCP project | `cid-hks-1537286359734` (`us-east4`) |
| GCS bucket | `gs://gl-deep-search-data` |

> ⚠️ **Two cluster events are imminent — check both before starting an 18-hour job.** Per `~/ln_gl/openalex/src/fasrc_cluster_guide.md`:
>
> - **FASRC rolling OS upgrades, Aug 24–27 2026** — three days from now. Don't have a long job in flight across that window.
> - **`/n/holystore01` migrates off legacy Tier 0 Lustre in late September 2026** (Storage Modernization Initiative, Phase 4) — roughly five weeks out. **Every path in this document lives on it**, and `PROJECT_DIR` is hardcoded at `deployment/slurm/etl_pipeline.sbatch:44`. Confirm destination paths and timing for `hausmann_lab` with `rdm@rc.fas.harvard.edu`.
>
> Practically: the Phase 2 re-run wants to happen either in the next three days or in the window between the OS upgrade and the storage migration. Paths verified working 2026-08-21.

---

## Staged rollout, 2026-08-21

Run as three gates, each of which caught something the previous one could not.

### Gate 1 — rebuild the image (4 attempts)

| # | Result | What it taught |
|---|---|---|
| 1 | FAILURE | `uv.lock`'s `[options]` block breaks `uv sync --locked` in the container |
| 2 | SUCCESS (13m44s) | `--frozen` fixes it; but this image predated the chunk-filter commit |
| 3 | SUCCESS (15m1s) | Image imported fine by `python -c`, yet **failed the real job invocation** |
| 4 | SUCCESS | `chmod -R a+rX` in the builder; verified against the exact `singularity exec --pwd /app` path |

### Gate 2 — cluster smoke test (job `40965304`)

`SOURCES=all SCRAPER_LIMIT=5 DOWNLOAD_LIMIT=4 PDF_LIMIT=4 TRANSCRIPTS_LIMIT=1`

**COMPLETED in 14m38s, 0 errors, all 8 components.** Two things ran on the cluster for the first time ever: the **OpenAlex path** (332 publications scraped, 2 OA files downloaded) and **lecture transcripts** (1/1). 5 documents → 347 chunks → 347 embeddings, all written back to persistent storage.

The staleness guard printed exactly as designed:

```
Repo commit:   de66b652d697634ee1817f93a156be83d85a3ad7
Image commit:  de66b652d697634ee1817f93a156be83d85a3ad7
Image matches checkout.
```

**The smoke test also confirmed why the embedding fix is load-bearing.** The log read `Embedding 5 documents (0 from tracker, 5 discovered on disk)`. Querying the tracker explains the zero:

| | count |
|---|---|
| `processing_status = PROCESSED` | 331 |
| `embedding_status = PENDING` | 123 |
| **intersection (what the tracker query returns)** | **0** |

The 331 processed rows are exactly the 293 `EMBEDDED` + 38 `FAILED`. The tracker will *never* return them, so **only disk discovery can reach those 331 documents**. Without the union fix they would have been chunked and then silently dropped. Verified no rows were flipped to `FAILED` by the run (38 before, 38 after).

### Gate 3 — full production run (job `40971461`)

Submitted `SOURCES=all` with no limits, 48h limit, tracking DB backed up first. Sized from measured evidence:

| Stage | Measured | Extrapolated |
|---|---|---|
| PDF extraction | 145s/PDF on A100 (142s in Feb, 168s incl. model load today) | ~520 PDFs → **~21h** |
| Embeddings | 347 chunks in 23s (~15/s) | ~36,700 chunks → **~45min** |
| OpenAlex downloads | 332 publications, ~50% OA hit rate | ~1–2h |
| **Total** | | **~23–24h** |

Hence 48h rather than 24h. `--mem` deliberately left at 100G despite a 4.96G MaxRSS: that evidence is from 4 small PDFs and is not representative, and on a 4-GPU-per-node partition the GPU count caps packing long before 4×100G does — so trimming it would buy nothing. Re-check with `jobstats` once a full run finishes.

---

## Deployment gotchas found the hard way (2026-08-21)

Four separate failures sat between "code is correct" and "job runs", none visible from the source. Recorded so the next gap doesn't rediscover them.

**1. `uv.lock` carries a developer's global config.** A global `~/.config/uv/uv.toml` with `exclude-newer = "7 days"` makes uv write an `[options]` block into `uv.lock`. A config-less uv inside the container reads that block, decides the lockfile is invalid, re-resolves, and `--locked` then fails the build outright:

```
Ignoring existing lockfile due to removal of global exclude newer
The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.
```

Image builds now use `--frozen` (install the pins, don't re-validate them); CI keeps `--locked` so the lock is still checked somewhere. Locally, use `uv --no-config run --frozen ...` to avoid rewriting the lock. **Committing a lock with an `[options]` block breaks CI**, so check `grep -c '^\[options\]' uv.lock` returns 0 before committing.

**2. Build-context permissions break Singularity but not Docker.** The repo now lives under `~/Library/CloudStorage/Dropbox-Personal/...` (Dropbox's CloudStorage migration), whose directories are mode `0700`. `COPY . /app` preserves that, so `/app/backend` landed in the image as `drwx------ root`. Docker builds and runs as root and never noticed. Singularity runs as the invoking user, so the directory was untraversable and the exact job invocation died instantly:

```
singularity exec --pwd /app ... python -m backend.etl.orchestrator
ModuleNotFoundError: No module named 'backend.etl'
```

The relative `--config backend/etl/config.yaml` path was unreadable for the same reason. Fixed with `chmod -R a+rX /app` in the **builder** stage — doing it after the runtime `COPY` would duplicate the ~4.6 GB virtualenv into a new layer. This is why the February image worked and a rebuild from identical code did not.

**3. Stale GitHub host key on the cluster.** `git pull` failed intermittently with "Please make sure you have the correct access rights" — which is not an auth problem at all. An old GitHub host key pinned to an IP in `~/.ssh/known_hosts` conflicted, and it only tripped when DNS resolved to that IP. Fixed with `ssh-keygen -R 140.82.112.4`. If it recurs for another IP, that's the same cause.

**4. Always verify the image by running the real invocation.** `python -c "import backend..."` passes even when the job would fail, because it resolves the package from the venv's site-packages. Only `singularity exec --writable-tmpfs --pwd /app --bind <data> ... python -m backend.etl.orchestrator --help` exercises the path the job takes. Note the data bind is required: importing the orchestrator initialises the SQLite tracker as a module-level side effect, so without it you get a misleading `unable to open database file`.

---

## Work done 2026-08-21

All local, all verified. Nothing has been committed, built, or submitted to the cluster.

| Change | Files |
|---|---|
| **bf16 model loading** — `SentenceTransformer` now receives `model_kwargs={"dtype": ...}`. Halves resident weights for the 8B model from ~32 GB to ~16 GB | `embeddings_generator.py`, `config.yaml` |
| **Sequence cap** — `max_seq_length: 2048`, and `max_chunk_size` lowered 8000 → 2000 tokens so outlier chunks stop driving quadratic attention cost | `embeddings_generator.py`, `config.yaml` |
| **OOM retry floor** — halving now continues to batch size 1 instead of stopping at 4 | `embeddings_generator.py` |
| **Per-document memory release** — `release_gpu_memory()` after every document, not only on OOM | `embeddings_generator.py` |
| **Safe defaults** — dtype/seq-length default to `None` so small CPU models are unaffected; production values live in `config.yaml`, dev overrides to `null` | `config.dev.yaml` |
| **Git SHA provenance** — commit baked into the image at build time and passed through Cloud Build | `pdf-processing/Dockerfile`, `cloudbuild-slurm.yaml`, `setup_env.sh` |
| **Staleness guard** — sbatch compares image SHA to checkout SHA and refuses to run on mismatch (`ALLOW_STALE_IMAGE=1` to override). Degrades to a warning when either SHA is unavailable | `etl_pipeline.sbatch` |
| **Time limit** 12h → 24h, with the 17h14m evidence recorded in the comment | `etl_pipeline.sbatch` |
| **3 regression tests** — retry reaches batch size 1; non-OOM `RuntimeError` propagates unretried; memory controls actually reach the model | `test_embeddings_generator.py` |

Verification: **282 passed, 3 skipped**; `ruff check`, `ruff format --check`, `mypy` all clean; `--dev --sources all --dry-run` sequences all 8 components correctly.

Two notes worth keeping:

- The kwarg is **`dtype`, not `torch_dtype`** — transformers 4.57.6 (the pinned version) deprecates the latter with a `FutureWarning`.
- The dry run confirms the pipeline ends at *Embeddings Generator*. **There is no ingestion component.** That is the missing link between ETL and search, and it is the real remaining work.

---

## The one-paragraph version

The February work was good and it is not lost. The ETL pipeline is proven at production scale, the test suite is green, and the search layer is written. What killed momentum was a stale container image producing a two-minute failure at 9 PM, followed by six months of not coming back to it. Rebuild the image, fix the embedding OOM before spending GPU hours, re-run the pipeline, and then do the one genuinely unfinished thing: stand up Qdrant and ingest. That is the shortest path from here to a system that can answer a question.
