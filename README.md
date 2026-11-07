# Gendered AI Research

A reproducible research project for studying author representation across computer-science subfields. The project is organized into data ingestion, name-based inference, validation, and statistical analysis stages.

## Research and ethical scope

Names and public biographies are imperfect signals and do not establish a person's gender identity. Inferred labels are uncertain, may encode cultural and language biases, and should not be used to make decisions about individuals. Preserve `unknown` when evidence is insufficient, report coverage and error rates, and use human-reviewed data only with appropriate safeguards and consent.

## Setup

Python 3.10 or later is recommended.

When this folder is opened in VS Code, new integrated terminals automatically
activate the project environment in `.venv`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `config.yaml` for the study period, categories, and thresholds. The pipeline
reads `GENDERIZE_API_KEY` from the process environment. VS Code terminals are
configured to load `.env`; in other environments, load the file or set the
variable yourself. Do not commit `.env` or collected personal data.

The configured taxonomy uses compound arXiv category/title/abstract queries for
11 granular subfields. Each field records its macro cluster, target scope, and
representation hypothesis. The collector still accepts the legacy list of arXiv
categories for older configurations. Keyword-based fields are operational
approximations rather than citation-network topics; papers can match more than one
subfield, so comparisons should not treat field membership as mutually exclusive.
The default target is 800 papers per subfield. A warning is emitted when a field
has fewer than 350 authors with a known inferred gender, and coverage is reported
alongside the representation estimates.

## Project layout

- `data/raw/`: reserved for immutable source responses and roster exports
- `data/interim/`: normalized paper metadata and author lists
- `data/processed/`: validated, analysis-ready CSV files
- `notebooks/`: exploratory, quality-control, and modeling workflows
- `src/`: reusable ingestion, inference, validation, analysis, and utility modules
- `tests/`: unit tests for parsing and statistical helpers
- `figures/`: exported plots

The static-frontend datasets can be generated from the pipeline CSVs with
`python -m src.export.parquet_builder`. This writes Snappy-compressed
`authors_enriched.parquet` and `papers.parquet` files to
`frontend/public/data/` using typed Arrow schemas and dictionary-encoded
subfield and country columns. Pass `--append` to merge a new batch with both
existing files; overlapping papers and author records are replaced by the
incoming rows. The builder validates gender labels and reports the
unknown-label rate for each subfield; it fails if
`authors_enriched.parquet` is not below 15,000,000 bytes. Without verified
OpenAlex enrichment, country, academic age, and career cohort remain
`UNKNOWN`, `-1`, and `unknown` rather than being inferred.

## Static frontend

The Vite application in `frontend/` initializes DuckDB-WASM in a Web Worker and
loads both Parquet datasets into temporary DuckDB tables for browser-side SQL
queries. The files are loaded from `frontend/public/data/`; when deployed,
DuckDB's HTTP reader can fetch the static files directly, including byte-range
reads. Startup materializes compact chart and filter summaries to avoid
re-aggregating all author rows for each visualization. The dashboard provides
macro-cluster/subfield, affiliation country,
cohort, gender, and publication-year filters; subfield comparison,
first/last-author comparison, classification-coverage, and verified-country
visualizations; and a paginated paper browser with filtered CSV export. The grid
displays one row per paper, while the CSV includes matching author records and
paper metadata.

Representation percentages use female and male labels as the classified
denominator; non-binary and unknown labels remain separate in the coverage
visualization. Gender selection applies to the paper browser and CSV only, not
to representation denominators. First/last-author differences are descriptive:
the current dataset does not provide significance-test results. Country charts
omit unknown country codes, which should not be read as an absence of authors
from those locations. The current pipeline may not contain enriched country or
career-cohort values.

Run the frontend from its directory with `npm install`, `npm run dev`, or
`npm run build`. Filters use empty arrays to mean “all values”; `yearRange` is
an inclusive pair of integer publication years.

The GitHub Pages workflow runs on pushes to `main` and monthly at
`0 0 1 * *`. It tests the Python pipeline, processes the previous 30 days of
arXiv records, appends the batch to the Parquet datasets, persists those files
to `main`, builds the frontend, and deploys to GitHub Pages.

## How data is collected

The pipeline collects paper metadata and author display names from the public
arXiv Atom API; it does not discover author biographies or personal web pages.
For each configured subfield, it combines the subfield's arXiv category/title/
abstract query with the configured date window. The API query uses the submitted-
date range, and returned records are checked against the configured dates before
use. The default window starts on 2020-01-01 and uses `present` as a dynamic end
date. Parsed papers retain the published date and primary category, and each
author record is tagged as first, middle, or last in the arXiv author list
(single-author papers are tagged first).

Results are paginated, deduplicated by paper ID within each subfield, and capped
at `sample_size_per_subfield` (800 by default). This is a cap on the first unique
matching results returned by arXiv, **not a randomized sample**. A paper that
matches multiple subfields can appear in each one. The client waits at least
three seconds between requests and retries temporary HTTP 503 responses. If a
subfield returns fewer papers than requested, the pipeline logs a warning rather
than filling the sample from another source.

OpenAlex enrichment is available but disabled by default to keep ordinary
pipeline runs offline and avoid unexpectedly large external API workloads. Set
`enrichment.enabled: true` to look up papers by DOI or arXiv identifier, with an
exact-title fallback, and match author names to that paper's authorships. The
first paper-specific institutional country code is preferred; if unavailable,
an author email's country-code TLD may be used, while generic email domains and
unresolved cases remain `UNKNOWN`. Academic age uses OpenAlex's
`first_publication_year`, falling back to an exact-name arXiv author search
sorted from earliest submission. Cohorts are junior (0-3 years), mid-career
(4-7), and senior (8+). Emails are used only to derive a country TLD and are
not written to exported author rows. HTTP 404 and 429 responses fall back to
unknown metadata; other HTTP failures are surfaced.

The same opt-in enables pronoun extraction only for verified profile URLs
explicitly supplied by a caller through `run_pipeline(author_profile_urls=...)`,
keyed by author `record_id` with optional `faculty_bio_url` and
`personal_homepage_url` or `csrankings_profile_url` values. The faculty bio,
CSRankings profile, and homepage are checked in that order. Explicit pronoun
pairs (such as `she/her`) or a repeated unambiguous pronoun signal can take
precedence as a Tier 1 label (including `non_binary`). Otherwise, the
probabilistic Tier 2 gates apply and failed classifications remain `unknown`.
The resolver fetches only supplied URLs and does not discover profile links.

Paper metadata and author lists are normalized to
`data/interim/arxiv_papers.csv`. Each non-empty author name becomes a separate
record in `data/processed/arxiv_authors_final.csv`, with its paper, subfield,
author position, and a stable record ID. The pipeline does not currently save
the original API XML responses.

## How names are classified

Classification is an estimate based on an author's given name, not a claim about
their gender identity:

1. The pipeline conservatively parses the display name, removes recognized
   titles and suffixes, and uses the first non-initial name token as the given
   name. Names without a usable given name remain `unknown`.
2. The local `gender-guesser` dictionary is applied first. Clear `female` or
   `male` matches meet the configured confidence threshold; weaker or
   inconclusive results do not.
3. If OpenAlex enrichment is enabled, a caller-verified profile may provide
   explicit pronouns for Tier 1. Otherwise, Tier 2 accepts a probabilistic label
   only when confidence is at least 0.95, its sample count is at least 100, and
   the configured country guard passes. Name-only local
   results without a qualifying sample count remain `unknown`.
4. Genderize.io is called only when `GENDERIZE_API_KEY` is set. Without the key,
   uncertain local results remain `unknown` and names are not sent to the
   service.

The output labels include `female`, `male`, `non_binary` (explicit pronoun
evidence only), and `unknown`. Each output row records the inference source,
confidence, validation tier, and pronoun evidence. Repeated given names are
inferred once per pipeline run and reused. Use results only for aggregate
research, and report unknown-name coverage alongside representation estimates.

For review, the pipeline creates a reproducible audit queue stratified by
subfield, inferred label, and author position. It samples 10% per stratum by
default (at least one record in each non-empty stratum). A reviewer can enter a
label, review date, and notes in
`data/processed/manual_audit_ground_truth.csv`; completed reviews are retained
when the pipeline is rerun. Only reviewed records are marked `is_audited` in
the author output. These human labels support quality checks; they do not turn
automated name estimates into verified identity data.

## Typical workflow

Run the end-to-end workflow from the project root:

```powershell
python main.py
```

The pipeline writes aggregate summaries and figures in addition to the data
outputs described above. Use `python main.py --no-plots` to skip chart
generation, or `python main.py --config path\to\config.yaml` to select another
config.

The pipeline writes intermediate papers to `data/interim/`, processed author and
review data to `data/processed/`, and figures to `figures/`. Run the notebooks to
explore coverage, uncertainty, and representation estimates. The faculty summary is
computed from `data/processed/faculty_final.csv` when present; the pipeline does not
crawl faculty biographies or personal web pages.

The pipeline exports four figures as PNG and PDF: first-to-last-author dumbbells
(`pipeline_dropoff`), last-author odds ratios with 95% Wald intervals
(`odds_ratio_forest`), first-author inferred-gender and unknown-name coverage
(`author_coverage`), and first/last-author plus faculty point ranges
(`career_stage_comparison`). Point ranges show one binomial standard error. All
representation percentages use only records with known inferred gender; the coverage
figure separately shows the unknown share. Faculty points appear only when
`faculty_final.csv` contains known-gender records with matching subfield labels.

In addition to per-subfield summaries, `data/processed/macro_cluster_contrast.csv`
reports a pooled two-proportion z-test comparing first-author female representation
in the applied/socio-technical and core/infrastructure cohorts. The bridge cohort
is kept separate and excluded from that contrast.

All external collection should comply with source terms, rate limits, privacy requirements, and institutional review requirements. Do not scrape or infer information from private accounts.

## Configuration and credentials

`config.yaml` contains study parameters. `.env.example` documents optional service
credentials; load them in your execution environment rather than storing secrets
in source control. The core library functions do not make network requests merely
by being imported.

## Tests

```powershell
python -m pytest
```

## Processed author schema

The CSV header in `data/processed/arxiv_authors_final.csv` defines the expected output
columns, including a stable `record_id` used to connect audit entries to author records.
Use `unknown` when classification is unsupported, retain confidence and source fields,
and set `is_audited` only for records actually reviewed by a person.
