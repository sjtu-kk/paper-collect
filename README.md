# paper-collect

`paper-collect` is a topic-only live paper search runtime. It turns a research topic into staged, reviewable search artifacts:

1. `stage-01`: human-confirmed topic intent
2. `stage-02`: topic profile and boundaries
3. `stage-03`: provider-agnostic search intents
4. `stage-04`: live provider search, normalized candidates, BibTeX, and match ledger
5. `stage-05`: citation verification and verified BibTeX

The current implementation focuses on search evidence and reviewability. It does not download PDFs, retrieve full text, rank final relevance, use paid or institutional connectors, or perform exact-title acquisition.

## Providers

The active Stage 04 search providers are:

- OpenAlex
- Semantic Scholar
- arXiv
- Crossref
- DBLP
- DOAJ
- PubMed

Provider failures such as rate limits, timeouts, zero results, or network errors are written as structured evidence instead of being hidden.

## Install

```bash
python -m pip install -e .
```

Optional LLM planning uses an OpenAI-compatible endpoint. Copy `.env.example` to `.env` and fill values locally:

```bash
cp .env.example .env
```

Do not commit `.env`.

## Run

Start a topic-search run:

```bash
paper-collect topic-search \
  --topic "AI agents for autonomous research" \
  --description "Automated scholarly research agents for literature discovery and synthesis; exclude automotive/self-driving vehicle research." \
  --run-id ai-agents-demo \
  --max-results-per-query 5 \
  --year-min 2023 \
  --inter-query-delay 3 \
  --inter-verify-delay 1
```

The first command writes `runtime_data/topic_search_runs/<run-id>/stage-01/topic_intent.md` and stops until a human changes:

```yaml
status: draft
```

to:

```yaml
status: confirmed
```

Then resume:

```bash
paper-collect topic-search --resume --run-id ai-agents-demo
```

Render the offline review page:

```bash
python scripts/render_topic_search_review.py runtime_data/topic_search_runs/ai-agents-demo
```

Open `runtime_data/topic_search_runs/ai-agents-demo/review.html` in a browser.

## Test

```bash
python -m pytest tests -q
```

## Notes

- `stage-05` verifies bibliographic identity. It does not prove topic relevance.
- Search quality depends heavily on Stage 01 intent and Stage 03 query planning.
- Live provider behavior can vary because public APIs can rate-limit or return transient errors.

