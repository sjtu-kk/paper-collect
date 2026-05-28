from __future__ import annotations

import json
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from paper_collect import provider_registry
from paper_collect.literature import search
from paper_collect import topic_search
from topic_intent_helpers import write_confirmed_topic_intent


def test_active_provider_ids_are_seeded_from_the_registry() -> None:
    assert provider_registry.active_provider_ids() == (
        "openalex",
        "semantic_scholar",
        "arxiv",
        "crossref",
        "dblp",
        "doaj",
        "pubmed",
    )


def test_registry_entries_expose_decoupled_top_level_schema() -> None:
    entry = provider_registry.provider_entry("openalex")

    assert entry.registry_state == "active"
    assert entry.roles == ("broad_metadata_discovery", "venue_authority", "cross_domain_discovery")
    assert entry.venue_coverage.kind == "entity"
    assert entry.venue_coverage.searchable is True
    assert entry.venue_coverage.corpus_limited is False
    assert entry.supported_filters.implemented == ("year_min",)
    assert entry.supported_filters.documented_reference == ("venue", "doi", "open_access", "publisher")
    assert entry.supported_filters.future_candidates == ("category", "language", "date_range")
    assert entry.access_model.tags == ("public", "email_required", "rate_limited", "network_risk")


def test_openalex_registry_declares_source_authority_evidence_fields() -> None:
    entry = provider_registry.provider_entry("openalex")

    assert {"source_id", "source_display_name", "source_issn", "source_type"}.issubset(entry.result_fields)
    assert "source_authority" in entry.provenance_capabilities
    assert "primary_location.source" in entry.supported_search_surfaces
    assert "source_authority_visibility" in entry.stage_usage


def test_registry_venue_coverage_examples_include_real_reviewable_venues() -> None:
    examples = {
        example
        for entry in provider_registry.provider_registry_entries()
        for example in entry.venue_coverage.examples
    }

    assert {
        "ACL",
        "CVPR",
        "AAAI",
        "NeurIPS",
        "JMLR",
        "Nature",
        "PLOS ONE",
        "JAMA",
    }.issubset(examples)
    assert "ICLR" not in examples


def test_registry_represents_reference_deferred_and_probe_states_without_expanding_active_set() -> None:
    assert provider_registry.active_provider_ids() == (
        "openalex",
        "semantic_scholar",
        "arxiv",
        "crossref",
        "dblp",
        "doaj",
        "pubmed",
    )
    assert provider_registry.provider_entry("opencitations").registry_state == "deferred"
    assert provider_registry.provider_entry("pid_repository_citation_graph_family").registry_state == "needs_live_probe"


def test_crossref_registry_declares_stage04_authority_behavior() -> None:
    entry = provider_registry.provider_entry("crossref")

    assert entry.registry_state == "active"
    assert entry.roles == ("bibliographic_authority", "doi_resolution", "venue_authority")
    assert entry.venue_coverage.kind == "entity"
    assert entry.venue_coverage.searchable is True
    assert entry.supported_search_surfaces == ("works.search",)
    assert entry.supported_filters.implemented == ("year_min",)
    assert {"container_title", "issn", "publisher", "crossref_id"}.issubset(entry.result_fields)
    assert "bibliographic_authority" in entry.provenance_capabilities
    assert "stage_04" in entry.stage_usage
    assert "authority_visibility" in entry.stage_usage
    assert entry.access_model.tags == ("public", "rate_limited", "network_risk")


def test_dblp_registry_declares_stage04_cs_bibliography_behavior() -> None:
    entry = provider_registry.provider_entry("dblp")

    assert entry.registry_state == "active"
    assert entry.roles == ("cs_bibliographic_authority", "venue_authority")
    assert entry.coverage_scope == ("computer_science",)
    assert entry.venue_coverage.kind == "entity"
    assert entry.venue_coverage.corpus_limited is True
    assert entry.supported_search_surfaces == ("publication.search",)
    assert entry.supported_filters.implemented == ("year_min",)
    assert {"dblp_key", "venue", "publication_type", "ee"}.issubset(entry.result_fields)
    assert "cs_bibliographic_authority" in entry.provenance_capabilities
    assert "stage_04" in entry.stage_usage
    assert "cs_bibliography_visibility" in entry.stage_usage
    assert entry.access_model.tags == ("public", "rate_limited", "network_risk")


def test_doaj_registry_declares_stage04_oa_metadata_behavior() -> None:
    entry = provider_registry.provider_entry("doaj")

    assert entry.registry_state == "active"
    assert entry.roles == ("oa_journal_discovery", "oa_article_metadata", "venue_authority")
    assert entry.coverage_scope == ("open_access_journals", "open_access_articles")
    assert entry.venue_coverage.kind == "entity"
    assert entry.venue_coverage.corpus_limited is True
    assert entry.supported_search_surfaces == ("articles.search",)
    assert entry.supported_filters.implemented == ("year_min",)
    assert {"doaj_id", "journal_title", "issn", "publisher", "license", "language"}.issubset(entry.result_fields)
    assert "oa_metadata" in entry.provenance_capabilities
    assert "stage_04" in entry.stage_usage
    assert "oa_metadata_visibility" in entry.stage_usage
    assert entry.access_model.tags == ("public", "rate_limited", "network_risk")


def test_pubmed_registry_declares_stage04_biomedical_corpus_behavior() -> None:
    entry = provider_registry.provider_entry("pubmed")

    assert entry.registry_state == "active"
    assert entry.roles == ("biomedical_corpus", "bibliographic_authority")
    assert entry.coverage_scope == ("biomedical_literature",)
    assert entry.venue_coverage.kind == "returned_only"
    assert entry.venue_coverage.corpus_limited is True
    assert entry.supported_search_surfaces == ("esearch", "esummary")
    assert entry.supported_filters.implemented == ("year_min",)
    assert {"pmid", "doi", "journal", "publication_types", "language", "has_abstract"}.issubset(entry.result_fields)
    assert "biomedical_corpus" in entry.provenance_capabilities
    assert "stage_04" in entry.stage_usage
    assert "biomedical_corpus_visibility" in entry.stage_usage
    assert entry.access_model.tags == ("public", "email_required", "token_recommended", "rate_limited", "network_risk")


def test_registry_validation_rejects_invalid_access_model_category() -> None:
    invalid_entry = replace(
        provider_registry.provider_entry("openalex"),
        access_model=provider_registry.AccessModel(
            tags=("invalid_category",),
        ),
    )

    with pytest.raises(ValueError, match="access_model"):
        provider_registry.validate_provider_registry_entry(invalid_entry)


def test_registry_access_model_distinguishes_required_access_states() -> None:
    tags = {
        tag
        for entry in provider_registry.provider_registry_entries()
        for tag in entry.access_model.tags
    }

    assert {
        "public",
        "api_key_required",
        "email_required",
        "token_recommended",
        "paid_or_institutional",
        "network_risk",
    }.issubset(tags)


def test_registry_validation_rejects_missing_required_decoupled_fields() -> None:
    invalid_entry = replace(provider_registry.provider_entry("openalex"), stage_usage=())

    with pytest.raises(ValueError, match="stage_usage"):
        provider_registry.validate_provider_registry_entry(invalid_entry)


def test_literature_search_reads_active_providers_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_registry, "active_provider_ids", lambda: ("openalex", "semantic_scholar"), raising=False)

    captured: dict[str, object] = {}

    def fake_search_source(source_name, query, *, limit, year_min, s2_api_key):  # noqa: ANN001
        captured.setdefault("sources", []).append(source_name)
        return {
            "papers": [],
            "status": {
                "source_name": source_name,
                "status": "completed",
                "returned_count": 0,
                "warnings": [],
            },
        }

    monkeypatch.setattr(search, "_search_source", fake_search_source, raising=False)

    result = search.search_papers("registry accessor runtime check")

    assert result.provider_statuses == [
        {"source_name": "openalex", "status": "completed", "returned_count": 0, "warnings": []},
        {"source_name": "semantic_scholar", "status": "completed", "returned_count": 0, "warnings": []},
    ]
    assert captured["sources"] == ["openalex", "semantic_scholar"]


def test_topic_search_stage03_reads_active_providers_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(provider_registry, "active_provider_ids", lambda: ("openalex", "semantic_scholar"), raising=False)
    monkeypatch.setattr(topic_search, "search_papers_multi_query", lambda *args, **kwargs: pytest.fail("stage 04 must not run in plan-only mode"), raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    output_root = tmp_path / "runs"
    write_confirmed_topic_intent(output_root / "runtime-registry", topic="registry accessor runtime check")
    result = topic_search.run_topic_search(
        topic=None,
        run_id="runtime-registry",
        output_root=output_root,
        resume=True,
        plan_only=True,
        allow_deterministic_fallback=True,
    )

    sources = json.loads((result.run_root / "stage-03" / "sources.json").read_text(encoding="utf-8"))

    assert sources["active_provider_ids"] == ["openalex", "semantic_scholar"]
    assert sources["reference_plan_labels"] == ["openreview", "google_scholar"]
    assert [source["source_name"] for source in sources["sources"] if source["status"] == "active"] == [
        "openalex",
        "semantic_scholar",
    ]


def test_topic_search_persists_canonical_provider_registry_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(topic_search, "search_papers_multi_query", lambda *args, **kwargs: pytest.fail("stage 04 must not run in plan-only mode"), raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    output_root = tmp_path / "runs"
    write_confirmed_topic_intent(output_root / "registry-snapshot", topic="registry snapshot contract")
    result = topic_search.run_topic_search(
        topic=None,
        run_id="registry-snapshot",
        output_root=output_root,
        resume=True,
        plan_only=True,
        allow_deterministic_fallback=True,
    )

    sources = json.loads((result.run_root / "stage-03" / "sources.json").read_text(encoding="utf-8"))
    snapshot = sources["provider_registry_snapshot"]
    entries = snapshot["provider_registry_entries"]
    openalex = next(entry for entry in entries if entry["provider_id"] == "openalex")

    assert snapshot["artifact_version"] == "paper_collect_provider_registry_snapshot.v1"
    assert [entry["provider_id"] for entry in entries if entry["registry_state"] == "active"] == [
        "openalex",
        "semantic_scholar",
        "arxiv",
        "crossref",
        "dblp",
        "doaj",
        "pubmed",
    ]
    assert openalex["evidence_refs"] == [
        "docs/research/paper-collect-venue-coverage-registry-decoupling.md",
    ]
    assert "evidence_references" not in json.dumps(snapshot)


def test_provider_registry_review_html_renders_real_venue_examples(tmp_path: Path) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "render_provider_registry_review.py"
    spec = importlib.util.spec_from_file_location("render_provider_registry_review", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = module.render(tmp_path / "provider-registry-review.html")
    html_text = output_path.read_text(encoding="utf-8")

    assert "老板视角的一句话" in html_text
    assert "当前能实际执行什么" in html_text
    assert "ACL" in html_text
    assert "CVPR" in html_text
    assert "AAAI" in html_text
    assert "NeurIPS" in html_text
    assert "ICLR" in html_text
    assert "不能硬写成已覆盖" in html_text
