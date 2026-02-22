# ADR-001: Self-hosted Marker for PDF parsing over hosted services

**Date:** 2026-02-20
**Status:** Accepted
**Decision:** Use self-hosted Marker (open-source, GPU-based) for PDF extraction instead of hosted APIs (LlamaParse, Datalab).

## Context

Our ETL pipeline extracts text from ~500 Growth Lab publication PDFs. We evaluated three options:

1. **Self-hosted Marker** (current approach) — open-source, runs on GPU
2. **LlamaParse** by LlamaIndex — hosted API, credit-based pricing
3. **Datalab** — hosted Marker API by the Marker authors

The question keeps resurfacing because self-hosted Marker requires GPU infrastructure (currently via Harvard FASRC SLURM cluster), which adds operational complexity. Hosted services would simplify the pipeline to a single API call per document.

## Cost Analysis

We sampled 22 PDFs by scraping 30 publications and downloading their files. Page counts were measured with `pypdf`.

### Sample statistics

| Metric | Value |
|---|---|
| PDFs sampled | 22 |
| Mean pages/PDF | 79.3 |
| Median pages/PDF | 50 |
| Std dev | 71.8 |
| Min / Max | 22 / 229 |
| 95% CI for mean | [47.5, 111.1] |

High variance is driven by a few large reports (220+ pages) alongside shorter working papers (20-50 pages).

### Projected total pages (500 PDFs)

| Metric | Pages |
|---|---|
| Point estimate | ~39,600 |
| 95% CI | [23,700 — 55,500] |

### LlamaParse pricing

1,000 credits = $1.25. Free tier: 10,000 credits/month.

| Tier | Credits/Page | $/Page | Est. Total Cost | 95% CI |
|---|---|---|---|---|
| Fast | 1 | $0.00125 | $49.55 | [$30, $69] |
| Cost Effective | 3 | $0.00375 | $148.64 | [$89, $208] |
| Agentic | 10 | $0.0125 | $495.45 | [$297, $694] |
| Agentic Plus | 45 | $0.05625 | $2,229.55 | [$1,335, $3,125] |

Free tier covers ~3,333 pages at Cost Effective (~42 PDFs of average length).

### Datalab (hosted Marker) pricing

Free trial: $5 in credits.

| Tier | $/1,000 pages | $/Page | Est. Total Cost | 95% CI |
|---|---|---|---|---|
| Fast | $4.00 | $0.004 | $158.55 | [$95, $222] |
| Balanced | $4.00 | $0.004 | $158.55 | [$95, $222] |
| High Accuracy | $6.00 | $0.006 | $237.82 | [$142, $333] |

Free trial covers ~1,250 pages at Balanced (~16 PDFs).

### Self-hosted Marker

$0 marginal cost. Requires GPU access (Harvard FASRC SLURM cluster, already available).

## Decision

**Self-hosted Marker wins on cost.** At recommended quality tiers, hosted services cost $149-$159 for a one-time run. But this is a recurring cost — every time we re-process (new publications, model upgrades, re-extraction after bugs), we pay again. Over multiple runs the cost compounds while self-hosted remains free.

The operational complexity of GPU access is manageable because:
- We already have SLURM infrastructure set up on FASRC
- PDF extraction is a batch job, not a latency-sensitive service
- Marker's extraction quality is identical to Datalab (same model)

### When to reconsider

- If we lose access to GPU infrastructure
- If the corpus grows significantly (thousands of PDFs) and self-hosted processing time becomes a bottleneck
- If hosted services introduce substantially cheaper tiers
- If we need features that only hosted services provide (e.g., LlamaParse's agentic mode for complex layouts)

## Methodology

Cost estimates were produced by a one-off script that:
1. Scraped 30 publications via `GrowthLabScraper`
2. Downloaded PDFs via `FileDownloader`
3. Counted pages with `pypdf`
4. Computed 95% confidence intervals using a t-distribution scaled to 500 PDFs
