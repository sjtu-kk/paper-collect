from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from paper_collect import cli
from paper_collect import provider_registry
from paper_collect import topic_search
from paper_collect.literature import Author, Paper, SearchBatchResult
from scripts import render_topic_search_review
from topic_intent_helpers import install_stage01_autoresume


@pytest.fixture(autouse=True)
def _isolate_real_workspace_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_workspace = tmp_path / "env-isolated-paper-collect"
    fake_package_dir = fake_workspace / "src" / "paper_collect"
    fake_package_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_package_dir / "cli.py"))
    monkeypatch.chdir(tmp_path)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    install_stage01_autoresume(monkeypatch, cli_module=cli, topic_search_module=topic_search)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _paper(*, paper_id: str = "p-1", title: str = "Search Only Paper", source: str = "openalex") -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=(Author(name="A. Author"),),
        year=2025,
        abstract=f"{title} abstract",
        venue="Test Venue",
        citation_count=3,
        doi="10.1000/search-only",
        url=f"https://example.org/{paper_id}",
        source=source,
    )


def _patch_stage04_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    papers: list[Paper] | None = None,
    provider_statuses: list[dict] | None = None,
) -> dict:
    captured: dict = {}
    papers = papers or []

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:
        captured["queries"] = list(queries)
        captured["kwargs"] = dict(kwargs)
        return SearchBatchResult(
            queries_used=list(queries),
            papers=list(papers),
            provider_statuses=provider_statuses or [
                {
                    "source_name": "openalex",
                    "status": "completed",
                    "returned_count": len(papers),
                    "queries_attempted": len(queries),
                    "warnings": [],
                },
                {
                    "source_name": "semantic_scholar",
                    "status": "completed",
                    "returned_count": 0,
                    "queries_attempted": len(queries),
                    "warnings": [],
                },
                {
                    "source_name": "arxiv",
                    "status": "completed",
                    "returned_count": 0,
                    "queries_attempted": len(queries),
                    "warnings": [],
                },
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)
    return captured


def _embedded_review_payload(review_html: str) -> dict:
    marker = '<script id="review-data" type="application/json">'
    start = review_html.index(marker) + len(marker)
    end = review_html.index("</script>", start)
    return json.loads(review_html[start:end])


@contextmanager
def _fake_llm_response(monkeypatch: pytest.MonkeyPatch, payload: dict | str):  # noqa: ANN001
    captured: dict = {}

    class FakeResponse:
        def __init__(self, content: dict | str) -> None:
            self._content = content

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            content = self._content if isinstance(self._content, str) else json.dumps(self._content)
            response = {
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            }
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        body = json.loads(request.data.decode("utf-8"))
        user_prompt = body["messages"][1]["content"]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = body
        captured["headers"] = dict(request.header_items())
        if "Stage 02-lite topic profile" in user_prompt and not (
            isinstance(payload, dict) and "topic_profile" in payload
        ):
            return FakeResponse(
                {
                    "topic_profile": {
                        "topic_goal": "Find papers about autonomous AI research agents for biomedical work",
                        "scope": ["software agents", "biomedical literature discovery"],
                        "boundary_notes": ["exclude human staffing agencies"],
                        "disambiguation_hints": ["agent means software agent"],
                        "negative_terms": ["real estate agents"],
                        "sub_questions": ["How do agents search biomedical literature?"],
                        "priorities": ["recent survey papers"],
                        "review_hints": ["check citation verification evidence"],
                    }
                }
            )
        return FakeResponse(payload)

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fake_urlopen)
    yield captured


@contextmanager
def _fake_stage02_and_stage03(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage02_profile: dict,
    stage03_plan: dict,
):  # noqa: ANN001
    class FakeResponse:
        def __init__(self, content: dict) -> None:
            self._content = content

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            response = {"choices": [{"message": {"content": json.dumps(self._content)}}]}
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        body = json.loads(request.data.decode("utf-8"))
        user_prompt = body["messages"][1]["content"]
        if "Stage 02-lite topic profile" in user_prompt:
            return FakeResponse({"topic_profile": stage02_profile})
        if "Stage 03 literature-search query plan" in user_prompt:
            return FakeResponse(stage03_plan)
        raise AssertionError(f"unexpected prompt: {user_prompt}")

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fake_urlopen)
    yield


def test_topic_search_cli_uses_deterministic_fallback_when_llm_missing_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "missing-llm-default",
            "--output-root",
            str(tmp_path / "runs"),
            "--plan-only",
        ]
    )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "missing-llm-default"
    topic_profile = _read_yaml(run_root / "stage-02" / "topic_profile.md")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    assert topic_profile["generation"]["mode"] == "deterministic_fallback"
    assert "quality_warnings" in topic_profile
    assert search_plan["query_generation"]["mode"] == "deterministic_fallback"


def test_search_only_cli_skips_stage05_and_review_marks_verification_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(monkeypatch, papers=[_paper()])
    monkeypatch.setattr(
        topic_search,
        "verify_citations",
        lambda *args, **kwargs: pytest.fail("Stage 05 verification must not run in search-only mode"),
        raising=False,
    )

    output_root = tmp_path / "runtime_data" / "topic_search_runs"
    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "search-only",
            "--output-root",
            str(output_root),
            "--max-results-per-query",
            "5",
            "--skip-verification",
            "--inter-query-delay",
            "0",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    run_root = output_root / "search-only"
    run_meta = _read_json(run_root / "run_meta.json")
    assert run_meta["stage_statuses"]["stage-01"] == "confirmed"
    assert run_meta["stage_statuses"]["stage-02"] == "complete"
    assert run_meta["stage_statuses"]["stage-03"] == "complete"
    assert run_meta["stage_statuses"]["stage-04"] == "complete"
    assert run_meta["stage_statuses"]["stage-05"] == "disabled"
    assert run_meta["verification"]["status"] == "disabled"
    assert not (run_root / "stage-05" / "verification_report.json").exists()

    review_path = render_topic_search_review.render(run_root)
    html = review_path.read_text(encoding="utf-8")
    assert '"status":"disabled"' in html
    assert "stage-05" in html


def test_search_only_review_surfaces_provider_visibility_without_stage05(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(
        monkeypatch,
        papers=[_paper(source="openalex")],
        provider_statuses=[
            {
                "source_name": "openalex",
                "status": "rate_limited",
                "returned_count": 1,
                "queries_attempted": 2,
                "warnings": ["429 rate limit"],
            },
            {
                "source_name": "semantic_scholar",
                "status": "missing_api_key",
                "returned_count": 0,
                "queries_attempted": 2,
                "warnings": ["api key required"],
            },
                {
                    "source_name": "arxiv",
                    "status": "network_failed",
                    "returned_count": 0,
                    "queries_attempted": 2,
                    "warnings": ["network unavailable"],
                },
                {
                    "source_name": "crossref",
                    "status": "provider_error",
                    "returned_count": 0,
                    "queries_attempted": 2,
                    "warnings": ["provider unavailable"],
                },
                {
                    "source_name": "dblp",
                    "status": "provider_error",
                    "returned_count": 0,
                    "queries_attempted": 2,
                    "warnings": ["DBLP unavailable"],
                },
                {
                    "source_name": "doaj",
                    "status": "provider_error",
                    "returned_count": 0,
                    "queries_attempted": 2,
                    "warnings": ["DOAJ unavailable"],
                },
                {
                    "source_name": "pubmed",
                    "status": "provider_error",
                    "returned_count": 0,
                    "queries_attempted": 2,
                    "warnings": ["PubMed unavailable"],
                },
            ],
        )
    monkeypatch.setattr(
        topic_search,
        "verify_citations",
        lambda *args, **kwargs: pytest.fail("Stage 05 verification must not run in search-only mode"),
        raising=False,
    )

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="review-search-only-visible-gaps",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    payload = _embedded_review_payload(review_path.read_text(encoding="utf-8"))

    assert payload["verification"]["status"] == "disabled"
    assert payload["evidence_sections"]["search"]["status"] == "present"
    assert payload["evidence_sections"]["verification"]["status"] == "disabled"
    assert payload["evidence_sections"]["search"]["artifact_refs"] == [
        "stage-04/search_meta.json",
        "stage-04/candidates.jsonl",
        "stage-04/match_ledger.jsonl",
        "stage-04/references.bib",
    ]
    assert payload["evidence_sections"]["verification"]["artifact_refs"] == []

    provider_ids = {entry["provider_id"] for entry in payload["provider_registry"]}
    assert {"openalex", "semantic_scholar", "arxiv", "crossref", "dblp", "doaj", "pubmed", "opencitations"}.issubset(provider_ids)
    openalex = next(entry for entry in payload["provider_registry"] if entry["provider_id"] == "openalex")
    assert openalex["venue_coverage"]["kind"] == "entity"
    assert openalex["access_model"]["tags"] == ["public", "email_required", "rate_limited", "network_risk"]
    assert "year_min" in openalex["supported_filters"]["implemented"]
    assert "venue" in openalex["supported_filters"]["documented_reference"]

    executions = payload["provider_executions"]
    assert payload["provider_execution_tasks"]
    assert [execution["status"] for execution in executions] == ["rate_limited", "auth", "network", "blocked", "blocked", "blocked", "blocked"]
    assert executions[0]["warnings"] == ["429 rate limit"]
    assert payload["review"]["gap_states"]["provider_execution_statuses"] == {
        "auth": 1,
        "blocked": 4,
        "network": 1,
        "rate_limited": 1,
    }
    assert "verification_disabled" in payload["review"]["gap_states"]["black_box_gaps"]


def test_review_payload_surfaces_active_provider_surface_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(
        monkeypatch,
        papers=[_paper(source="openalex")],
        provider_statuses=[
            {"source_name": "openalex", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
            {"source_name": "semantic_scholar", "status": "rate_limited", "returned_count": 0, "queries_attempted": 1, "warnings": ["rate limited"], "rate_control": {"provider_local": True}},
            {"source_name": "arxiv", "status": "rate_limited", "returned_count": 0, "queries_attempted": 1, "warnings": ["arXiv rate limited"], "rate_control": {"provider_local": True}},
            {"source_name": "crossref", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
            {
                "source_name": "dblp",
                "status": "completed_no_results",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": [],
                "compiled_query_summary": {
                    "original_query_text": "agentic literature review with provider prose",
                    "provider_query_text": "agentic literature review",
                    "rewrite_reason": "dblp_keyword_compaction",
                },
            },
            {"source_name": "doaj", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
            {"source_name": "pubmed", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
        ],
    )

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="review-provider-surface-audit",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    review_html = review_path.read_text(encoding="utf-8")
    visible_html = review_html.split('<script id="review-data"', 1)[0]
    payload = _embedded_review_payload(review_html)
    audit = payload["review"]["provider_surface_audit"]
    providers = {entry["provider_id"]: entry for entry in audit["providers"]}

    assert payload["provider_surface_audit"] == audit
    assert set(providers) == {"openalex", "semantic_scholar", "arxiv", "crossref", "dblp", "doaj", "pubmed"}
    assert providers["openalex"]["supported_search_surfaces"] == ["works.search", "title.search", "primary_location.source"]
    assert providers["openalex"]["executed_surface"] == "works.search"
    assert providers["dblp"]["coverage_judgment"] == "needs_provider_local_compaction"
    assert providers["dblp"]["surface_gap_state"] == "provider_local_compaction_applied"
    assert providers["pubmed"]["executed_surface"] == "esearch+esummary"
    assert "Provider surface audit" in visible_html
    assert "executed surface" in visible_html
    assert "supported_search_surfaces" in visible_html
    assert "matched_field" not in visible_html


def test_review_payload_surfaces_openalex_source_authority_contribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_metadata = {
        "evidence_kind": "openalex_source_authority",
        "source_id": "https://openalex.org/S4306400194",
        "display_name": "International Conference on Learning Representations",
        "issn_l": "",
        "issn": [],
        "type": "conference",
        "host_organization": "",
    }
    paper = Paper(
        paper_id="https://openalex.org/W-review-source",
        title="OpenAlex Review Source Authority",
        authors=(Author(name="A. Author"),),
        year=2026,
        abstract="review source evidence",
        venue="International Conference on Learning Representations",
        citation_count=3,
        doi="10.1000/review-source",
        url="https://openalex.org/W-review-source",
        source="openalex",
        source_metadata=source_metadata,
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[paper],
            provider_statuses=[
                {"source_name": "openalex", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
                {"source_name": "semantic_scholar", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "arxiv", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
            ],
            provider_result_rows=[
                {
                    "source_name": "openalex",
                    "query_text": queries[0],
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [paper.to_dict()],
                }
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="review-openalex-source-authority",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    review_html = review_path.read_text(encoding="utf-8")
    visible_html = review_html.split('<script id="review-data"', 1)[0]
    payload = _embedded_review_payload(review_html)

    assert payload["review"]["provider_source_authority"]["openalex"] == {
        "provider_id": "openalex",
        "target_effect": "source_authority_visibility",
        "boundary": "source metadata is authority/review evidence, not relevance ranking or full-text reachability",
        "returned_count": 1,
        "normalized_count": 1,
        "sources": [
            {
                "source_id": "https://openalex.org/S4306400194",
                "display_name": "International Conference on Learning Representations",
                "issn_l": "",
                "type": "conference",
            }
        ],
    }


def test_review_payload_surfaces_crossref_authority_contribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority_metadata = {
        "evidence_kind": "crossref_bibliographic_authority",
        "provider_id": "crossref",
        "crossref_id": "10.1145/1234567.8901234",
        "container_title": "Proceedings of the ACM Conference on Research Agents",
        "issn": ["1234-5678"],
        "publisher": "Association for Computing Machinery",
        "type": "proceedings-article",
    }
    paper = Paper(
        paper_id="10.1145/1234567.8901234",
        title="Crossref Review Authority",
        authors=(Author(name="A. Author"),),
        year=2024,
        abstract="crossref authority evidence",
        venue="Proceedings of the ACM Conference on Research Agents",
        citation_count=0,
        doi="10.1145/1234567.8901234",
        url="https://doi.org/10.1145/1234567.8901234",
        source="crossref",
        source_metadata=authority_metadata,
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[paper],
            provider_statuses=[
                {"source_name": "openalex", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "semantic_scholar", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "arxiv", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "crossref", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
            ],
            provider_result_rows=[
                {
                    "source_name": "crossref",
                    "query_text": queries[0],
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [paper.to_dict()],
                }
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="review-crossref-authority",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    review_html = review_path.read_text(encoding="utf-8")
    visible_html = review_html.split('<script id="review-data"', 1)[0]
    payload = _embedded_review_payload(review_html)

    assert payload["review"]["provider_source_authority"]["crossref"] == {
        "provider_id": "crossref",
        "target_effect": "journal_proceedings_bibliographic_authority",
        "boundary": "container-title, ISSN, and publisher metadata are authority/review evidence, not full-text reachability",
        "returned_count": 1,
        "normalized_count": 1,
        "records": [
            {
                "crossref_id": "10.1145/1234567.8901234",
                "container_title": "Proceedings of the ACM Conference on Research Agents",
                "issn": ["1234-5678"],
                "publisher": "Association for Computing Machinery",
                "type": "proceedings-article",
            }
        ],
    }
    assert "Crossref authority evidence" in visible_html
    assert "Proceedings of the ACM Conference on Research Agents" in visible_html
    assert "Association for Computing Machinery" in visible_html
    assert "container-title, ISSN, and publisher metadata are authority/review evidence, not full-text reachability" in visible_html


def test_review_payload_surfaces_dblp_cs_bibliography_contribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cs_metadata = {
        "evidence_kind": "dblp_cs_bibliographic",
        "provider_id": "dblp",
        "dblp_key": "conf/iclr/VelickovicCCRLB18",
        "venue": "ICLR",
        "publication_type": "Conference and Workshop Papers",
        "ee": "https://doi.org/10.48550/arXiv.1710.10903",
    }
    paper = Paper(
        paper_id="conf/iclr/VelickovicCCRLB18",
        title="Graph Attention Networks.",
        authors=(Author(name="Petar Velickovic"),),
        year=2018,
        abstract="",
        venue="ICLR",
        citation_count=0,
        doi="10.48550/arXiv.1710.10903",
        url="https://dblp.org/rec/conf/iclr/VelickovicCCRLB18",
        source="dblp",
        source_metadata=cs_metadata,
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[paper],
            provider_statuses=[
                {"source_name": "openalex", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "semantic_scholar", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "arxiv", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "crossref", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "dblp", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
            ],
            provider_result_rows=[
                {
                    "source_name": "dblp",
                    "query_text": queries[0],
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [paper.to_dict()],
                }
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="graph attention networks",
        run_id="review-dblp-cs-bibliography",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    review_html = review_path.read_text(encoding="utf-8")
    visible_html = review_html.split('<script id="review-data"', 1)[0]
    payload = _embedded_review_payload(review_html)

    assert payload["review"]["provider_source_authority"]["dblp"] == {
        "provider_id": "dblp",
        "target_effect": "cs_conference_bibliography_coverage",
        "boundary": "DBLP records are CS bibliography/conference metadata, not all-discipline authority or full-text reachability",
        "returned_count": 1,
        "normalized_count": 1,
        "records": [
            {
                "dblp_key": "conf/iclr/VelickovicCCRLB18",
                "venue": "ICLR",
                "publication_type": "Conference and Workshop Papers",
                "ee": "https://doi.org/10.48550/arXiv.1710.10903",
            }
        ],
    }
    assert "DBLP CS bibliography evidence" in visible_html
    assert "conf/iclr/VelickovicCCRLB18" in visible_html
    assert "ICLR" in visible_html
    assert "DBLP records are CS bibliography/conference metadata, not all-discipline authority or full-text reachability" in visible_html


def test_review_payload_surfaces_doaj_oa_metadata_contribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    oa_metadata = {
        "evidence_kind": "doaj_oa_metadata",
        "provider_id": "doaj",
        "doaj_id": "doaj-article-1",
        "journal_title": "Journal of Open Metadata",
        "issn": ["1234-5678"],
        "publisher": "Open Publisher",
        "license": ["CC BY"],
        "language": ["EN"],
        "link_url": "https://example.org/article-1",
        "link_type": "fulltext",
    }
    paper = Paper(
        paper_id="doaj-article-1",
        title="Open access graph attention networks",
        authors=(Author(name="Ada Author"),),
        year=2024,
        abstract="OA article metadata",
        venue="Journal of Open Metadata",
        citation_count=0,
        doi="10.1234/doaj.1",
        url="https://example.org/article-1",
        source="doaj",
        source_metadata=oa_metadata,
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[paper],
            provider_statuses=[
                {"source_name": "openalex", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "semantic_scholar", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "arxiv", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "crossref", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "dblp", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "doaj", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
            ],
            provider_result_rows=[
                {
                    "source_name": "doaj",
                    "query_text": queries[0],
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [paper.to_dict()],
                }
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="open access graph attention networks",
        run_id="review-doaj-oa-metadata",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    review_html = review_path.read_text(encoding="utf-8")
    visible_html = review_html.split('<script id="review-data"', 1)[0]
    payload = _embedded_review_payload(review_html)

    assert payload["review"]["provider_source_authority"]["doaj"] == {
        "provider_id": "doaj",
        "target_effect": "oa_journal_article_metadata_coverage",
        "boundary": "DOAJ records are OA journal/article metadata, not proof of acquired full text or PDF reachability",
        "returned_count": 1,
        "normalized_count": 1,
        "records": [
            {
                "doaj_id": "doaj-article-1",
                "journal_title": "Journal of Open Metadata",
                "issn": ["1234-5678"],
                "publisher": "Open Publisher",
                "license": ["CC BY"],
                "language": ["EN"],
                "link_url": "https://example.org/article-1",
                "link_type": "fulltext",
            }
        ],
    }
    assert "DOAJ OA metadata evidence" in visible_html
    assert "Journal of Open Metadata" in visible_html
    assert "CC BY" in visible_html
    assert "not proof of acquired full text or PDF reachability" in visible_html


def test_review_payload_surfaces_pubmed_biomedical_corpus_contribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    biomedical_metadata = {
        "evidence_kind": "pubmed_biomedical_corpus",
        "provider_id": "pubmed",
        "pmid": "42184189",
        "journal": "IEEE journal of biomedical and health informatics",
        "source": "IEEE J Biomed Health Inform",
        "publication_types": ["Journal Article"],
        "language": ["eng"],
        "has_abstract": True,
    }
    paper = Paper(
        paper_id="42184189",
        title="An Adaptive Fusion Network for Breast Tumor Grading Based on Graph Structure Learning.",
        authors=(Author(name="Yang L"),),
        year=2026,
        abstract="Has Abstract",
        venue="IEEE journal of biomedical and health informatics",
        citation_count=0,
        doi="10.1109/JBHI.2026.3696620",
        url="https://pubmed.ncbi.nlm.nih.gov/42184189/",
        source="pubmed",
        source_metadata=biomedical_metadata,
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[paper],
            provider_statuses=[
                {"source_name": "openalex", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "semantic_scholar", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "arxiv", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "crossref", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "dblp", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "doaj", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "pubmed", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
            ],
            provider_result_rows=[
                {
                    "source_name": "pubmed",
                    "query_text": queries[0],
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [paper.to_dict()],
                }
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="biomedical graph attention cancer",
        run_id="review-pubmed-biomedical-corpus",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    review_html = review_path.read_text(encoding="utf-8")
    visible_html = review_html.split('<script id="review-data"', 1)[0]
    payload = _embedded_review_payload(review_html)

    assert payload["review"]["provider_source_authority"]["pubmed"] == {
        "provider_id": "pubmed",
        "target_effect": "biomedical_corpus_coverage",
        "boundary": "PubMed records are biomedical corpus metadata, not all-discipline venue authority or full-text reachability",
        "returned_count": 1,
        "normalized_count": 1,
        "records": [
            {
                "pmid": "42184189",
                "journal": "IEEE journal of biomedical and health informatics",
                "source": "IEEE J Biomed Health Inform",
                "publication_types": ["Journal Article"],
                "language": ["eng"],
                "has_abstract": True,
            }
        ],
    }
    assert "PubMed biomedical corpus evidence" in visible_html
    assert "42184189" in visible_html
    assert "IEEE journal of biomedical and health informatics" in visible_html
    assert "not all-discipline venue authority or full-text reachability" in visible_html


def test_plan_only_review_renders_without_stage04_or_stage05(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        topic_search,
        "search_papers_multi_query",
        lambda *args, **kwargs: pytest.fail("Stage 04 must not run in plan-only mode"),
        raising=False,
    )
    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="review-plan-only",
        output_root=tmp_path / "runs",
        plan_only=True,
        allow_deterministic_fallback=True,
    )

    review_path = render_topic_search_review.render(result.run_root)
    payload = _embedded_review_payload(review_path.read_text(encoding="utf-8"))

    assert review_path.is_file()
    assert payload["search_meta"]["status"] == "disabled"
    assert payload["search_meta"]["total_candidates"] == 0
    assert payload["review"]["rows"] == []
    assert payload["provider_executions"] == []
    assert payload["evidence_sections"]["search"]["status"] == "disabled"
    assert payload["evidence_sections"]["search"]["artifact_refs"] == []
    assert payload["evidence_sections"]["verification"]["status"] == "disabled"
    assert payload["evidence_sections"]["verification"]["artifact_refs"] == []
    assert "search_disabled" in payload["review"]["gap_states"]["black_box_gaps"]
    assert "verification_disabled" in payload["review"]["gap_states"]["black_box_gaps"]


def test_review_uses_persisted_provider_registry_snapshot_not_live_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(monkeypatch, papers=[_paper(source="openalex")])
    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="review-stable-registry",
        output_root=tmp_path / "runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    def fail_live_registry() -> tuple:
        raise AssertionError("review renderer must use the run snapshot")

    monkeypatch.setattr(provider_registry, "provider_registry_entries", fail_live_registry, raising=False)

    review_path = render_topic_search_review.render(result.run_root)
    payload = _embedded_review_payload(review_path.read_text(encoding="utf-8"))

    assert payload["provider_registry_snapshot"]["artifact_version"] == "paper_collect_provider_registry_snapshot.v1"
    provider_ids = {entry["provider_id"] for entry in payload["provider_registry"]}
    assert {"openalex", "semantic_scholar", "arxiv"}.issubset(provider_ids)
    assert "fake_provider" not in provider_ids
    openalex = next(entry for entry in payload["provider_registry"] if entry["provider_id"] == "openalex")
    assert openalex["evidence_refs"] == [
        "docs/research/paper-collect-venue-coverage-registry-decoupling.md",
    ]
    assert "evidence_references" not in json.dumps(payload["provider_registry"])


def test_topic_search_cli_keeps_product_default_max_results_at_40(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _patch_stage04_search(monkeypatch, papers=[])

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "biomedical ai agents",
            "--run-id",
            "default-limit",
            "--output-root",
            str(tmp_path / "runs"),
            "--skip-verification",
            "--inter-query-delay",
            "0",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    assert captured["kwargs"]["limit_per_query"] == 40


def test_plan_only_cli_uses_valid_llm_plan_and_skips_provider_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        topic_search,
        "search_papers_multi_query",
        lambda *args, **kwargs: pytest.fail("Stage 04 must not run in plan-only mode"),
        raising=False,
    )
    llm_plan = {
        "search_strategies": [
            {
                "name": "core_methods",
                "intent_type": "core_concept",
                "queries": [
                    "biomedical research agents",
                    "clinical research agent systems",
                ],
                "preferred_sources": ["openalex", "semantic_scholar"],
                "negative_terms": ["real estate agents"],
                "disambiguation": "software agents for biomedical research, not human staffing agencies",
            }
        ],
        "filters": {"min_year": 2022},
    }

    with _fake_llm_response(monkeypatch, llm_plan) as captured:
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "llm-plan-only",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "llm-plan-only"
    run_meta = _read_json(run_root / "run_meta.json")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    sources = _read_json(run_root / "stage-03" / "sources.json")
    queries = _read_json(run_root / "stage-03" / "queries.json")

    assert captured["url"] == "https://llm.example.test/v1/chat/completions"
    assert captured["body"]["model"] == "test-model"
    stage03_prompt = captured["body"]["messages"][1]["content"]
    assert "Topic profile:" in stage03_prompt
    assert "real estate agents" in stage03_prompt
    assert "software agent" in stage03_prompt
    assert run_meta["stage_statuses"]["stage-01"] == "confirmed"
    assert run_meta["stage_statuses"]["stage-02"] == "complete"
    assert run_meta["stage_statuses"]["stage-03"] == "complete"
    assert run_meta["stage_statuses"]["stage-04"] == "disabled"
    assert run_meta["stage_statuses"]["stage-05"] == "disabled"
    assert run_meta["query_planning"]["mode"] == "llm"
    assert run_meta["query_planning"]["fallback_reason"] == ""
    assert not (run_root / "stage-03" / "search_plan.yaml").read_text(encoding="utf-8").lstrip().startswith("{")
    assert search_plan["query_generation"]["mode"] == "llm"
    assert search_plan["search_strategies"][0]["preferred_sources"] == ["openalex", "semantic_scholar"]
    assert search_plan["search_strategies"][0]["queries"] == [
        "biomedical research agents",
        "clinical research agent systems",
    ]
    assert queries["queries"][:2] == ["biomedical research agents", "clinical research agent systems"]
    assert len(search_plan["search_strategies"]) >= 3
    assert len(queries["queries"]) >= 8
    assert [source["source_name"] for source in sources["sources"] if source["status"] == "preferred"] == [
        "openalex",
        "semantic_scholar",
    ]
    assert not (run_root / "stage-04").exists()
    assert not (run_root / "stage-05").exists()


def test_plan_only_cli_writes_search_intents_as_primary_stage03_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm_plan = {
        "search_strategies": [
            {
                "name": "core_methods",
                "intent_type": "core_concept",
                "queries": [
                    "biomedical research agents",
                    "clinical research agent systems",
                ],
                "preferred_sources": ["openalex", "semantic_scholar"],
                "negative_terms": ["real estate agents"],
                "disambiguation": "software agents for biomedical research, not human staffing agencies",
            }
        ],
        "filters": {"min_year": 2022},
    }

    with _fake_llm_response(monkeypatch, llm_plan):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "llm-plan-intents",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "llm-plan-intents"
    run_meta = _read_json(run_root / "run_meta.json")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    sources = _read_json(run_root / "stage-03" / "sources.json")
    queries = _read_json(run_root / "stage-03" / "queries.json")
    intents = _read_json(run_root / "stage-03" / "search_intents.json")

    assert run_meta["stage_artifacts"]["stage-03"]["search_intents_json"] == "stage-03/search_intents.json"
    assert search_plan["topic_profile_ref"] == "../stage-02/topic_profile.md"
    assert search_plan["search_strategies"][0]["preferred_sources"] == ["openalex", "semantic_scholar"]
    assert queries["queries"][:2] == ["biomedical research agents", "clinical research agent systems"]
    assert len(search_plan["search_strategies"]) >= 3
    assert len(queries["queries"]) >= 8
    assert sources["active_provider_ids"] == ["openalex", "semantic_scholar", "arxiv", "crossref", "dblp", "doaj", "pubmed"]
    assert intents["artifact_version"] == "topic_search_search_intents.v1"
    assert intents["stage"] == "stage-03"
    assert intents["topic_profile_ref"] == "../stage-02/topic_profile.md"
    assert intents["filters"] in ({"year_min": 2022}, {})
    assert len(intents["intents"]) >= 8
    first_intent = intents["intents"][0]
    assert first_intent["intent_id"] == "intent_01"
    assert first_intent["intent_type"] == "core_concept"
    assert first_intent["query_text"] == "biomedical research agents"
    assert first_intent["source_strategy_ref"] == "search_strategies[0]"
    assert first_intent["filters"] in ({"year_min": 2022}, {})
    assert first_intent["rationale"]
    assert "provider_url" not in json.dumps(first_intent).lower()
    assert "http" not in json.dumps(first_intent).lower()
    assert "provider_url" not in json.dumps(intents).lower()
    assert not (run_root / "stage-04").exists()
    assert not (run_root / "stage-05").exists()


def test_stage04_consumes_search_intents_rather_than_flat_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm_plan = {
        "search_strategies": [
            {
                "name": "intent_contract",
                "intent_type": "core_concept",
                "queries": [
                    "intent contract query alpha",
                    "intent contract query beta",
                ],
                "preferred_sources": ["openalex"],
            }
        ],
        "filters": {"min_year": 2023},
    }

    captured = _patch_stage04_search(monkeypatch, papers=[_paper()])

    with _fake_llm_response(monkeypatch, llm_plan):
        result = topic_search.run_topic_search(
            topic="AI agents for biomedical research",
            run_id="intent-dominates-queries",
            output_root=tmp_path / "runs",
            max_results_per_query=5,
            year_min=2021,
            inter_query_delay=0.0,
            inter_verify_delay=0.0,
        )

    run_root = result.run_root
    search_intents = _read_json(run_root / "stage-03" / "search_intents.json")
    queries = _read_json(run_root / "stage-03" / "queries.json")
    search_meta = _read_json(run_root / "stage-04" / "search_meta.json")

    assert queries["queries"][:2] == ["intent contract query alpha", "intent contract query beta"]
    assert len(queries["queries"]) >= 8
    assert search_intents["intents"][0]["query_text"] == "intent contract query alpha"
    assert captured["queries"][:2] == ["intent contract query alpha", "intent contract query beta"]
    assert len(captured["queries"]) >= 8
    assert search_meta["query_source"] == "search_intents"
    assert search_meta["search_intents_contract_ref"] == "stage-03/search_intents.json"
    assert search_meta["queries_used"][:2] == ["intent contract query alpha", "intent contract query beta"]
    assert "flat-query" not in json.dumps(captured["queries"]).lower()


def test_plan_only_cli_accepts_llm_keyword_queries_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm_plan = {
        "search_strategies": [
            {
                "name": "sjtu_glm_shape",
                "intent_type": "core_concept",
                "keyword_queries": [
                    "LLM agents biomedicine",
                    "autonomous AI agents clinical research",
                ],
                "preferred_sources": ["semantic_scholar", "arxiv", "openalex"],
            }
        ],
        "filters": {"min_year": 2021},
    }

    with _fake_llm_response(monkeypatch, llm_plan):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "keyword-queries-alias",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "keyword-queries-alias"
    run_meta = _read_json(run_root / "run_meta.json")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    queries = _read_json(run_root / "stage-03" / "queries.json")
    assert run_meta["query_planning"]["mode"] == "llm"
    assert search_plan["search_strategies"][0]["queries"] == [
        "LLM agents biomedicine",
        "autonomous AI agents clinical research",
    ]
    assert queries["queries"][:2] == [
        "LLM agents biomedicine",
        "autonomous AI agents clinical research",
    ]
    assert len(search_plan["search_strategies"]) >= 3
    assert len(queries["queries"]) >= 8


def test_openai_compatible_planner_default_timeout_allows_slow_model_responses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_TIMEOUT_SECONDS", raising=False)
    llm_plan = {
        "search_strategies": [
            {
                "name": "slow_provider",
                "queries": ["biomedical research agents"],
                "preferred_sources": ["openalex"],
            }
        ]
    }

    with _fake_llm_response(monkeypatch, llm_plan) as captured:
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "default-timeout",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    assert captured["timeout"] == 120.0


def test_cli_loads_openai_compatible_config_from_workspace_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_workspace = tmp_path / "fake-paper-collect"
    fake_package_dir = fake_workspace / "src" / "paper_collect"
    fake_package_dir.mkdir(parents=True)
    (fake_workspace / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-key",
                "OPENAI_BASE_URL=https://env-file.example/v1",
                "OPENAI_MODEL=file-model",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "__file__", str(fake_package_dir / "cli.py"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    llm_plan = {
        "search_strategies": [
            {
                "name": "env_file",
                "queries": ["biomedical research agents"],
                "preferred_sources": ["openalex"],
            }
        ]
    }

    with _fake_llm_response(monkeypatch, llm_plan) as captured:
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "env-file-plan",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    assert captured["url"] == "https://env-file.example/v1/chat/completions"
    assert captured["body"]["model"] == "file-model"
    assert captured["headers"]["Authorization"] == "Bearer file-key"
    run_meta = _read_json(tmp_path / "runs" / "env-file-plan" / "run_meta.json")
    assert run_meta["query_planning"]["mode"] == "llm"


def test_invalid_llm_output_and_missing_env_fall_back_to_deterministic_stage03(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    with _fake_llm_response(monkeypatch, {"search_strategies": []}):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "invalid-llm",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    invalid_root = tmp_path / "runs" / "invalid-llm"
    invalid_meta = _read_json(invalid_root / "run_meta.json")
    invalid_plan = _read_yaml(invalid_root / "stage-03" / "search_plan.yaml")
    assert invalid_meta["query_planning"]["mode"] == "deterministic_fallback"
    assert invalid_meta["query_planning"]["fallback_reason"] == "invalid_llm_plan"
    assert invalid_plan["query_generation"]["mode"] == "deterministic_fallback"
    assert invalid_plan["query_generation"]["llm_required"] is False
    assert (invalid_root / "stage-03" / "sources.json").is_file()
    assert (invalid_root / "stage-03" / "queries.json").is_file()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "missing-env",
            "--output-root",
            str(tmp_path / "runs"),
            "--plan-only",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    missing_meta = _read_json(tmp_path / "runs" / "missing-env" / "run_meta.json")
    assert missing_meta["query_planning"]["mode"] == "deterministic_fallback"
    assert missing_meta["query_planning"]["fallback_reason"] == "missing_openai_env"


def test_empty_llm_output_falls_back_to_deterministic_stage03(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    with _fake_llm_response(monkeypatch, ""):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "empty-llm-output",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "empty-llm-output"
    run_meta = _read_json(run_root / "run_meta.json")
    assert run_meta["query_planning"]["mode"] == "deterministic_fallback"
    assert run_meta["query_planning"]["fallback_reason"] == "invalid_llm_plan"
    assert (run_root / "stage-03" / "search_plan.yaml").is_file()
    assert (run_root / "stage-03" / "sources.json").is_file()
    assert (run_root / "stage-03" / "queries.json").is_file()


def test_llm_call_failure_falls_back_to_deterministic_stage03(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    def fail_urlopen(*args, **kwargs):  # noqa: ANN001, ANN202
        raise OSError("network down")

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fail_urlopen)

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "llm-call-failure",
            "--output-root",
            str(tmp_path / "runs"),
            "--plan-only",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "llm-call-failure"
    run_meta = _read_json(run_root / "run_meta.json")
    assert run_meta["query_planning"]["mode"] == "deterministic_fallback"
    assert run_meta["query_planning"]["fallback_reason"] == "llm_call_failed"
    assert (run_root / "stage-03" / "search_plan.yaml").is_file()
    assert (run_root / "stage-03" / "sources.json").is_file()
    assert (run_root / "stage-03" / "queries.json").is_file()


def test_llm_plan_preferred_sources_are_bounded_to_registered_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm_plan = {
        "search_strategies": [
            {
                "name": "source_bounds",
                "queries": ["agent literature review"],
                "preferred_sources": [
                    "openalex",
                    "google_scholar",
                    "https://api.openalex.org/works?search=agent",
                    "arxiv",
                ],
                "provider_url": "https://api.example.test/search",
            }
        ]
    }

    with _fake_llm_response(monkeypatch, llm_plan):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "AI agents for biomedical research",
                "--run-id",
                "source-bounds",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    search_plan = _read_yaml(tmp_path / "runs" / "source-bounds" / "stage-03" / "search_plan.yaml")
    strategy = search_plan["search_strategies"][0]
    assert strategy["preferred_sources"] == ["openalex", "arxiv"]
    assert "provider_url" not in strategy
    assert not any("http" in json.dumps(strategy).lower() for _ in [0])


def test_llm_plan_preserves_disambiguation_negative_terms_and_query_hygiene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm_plan = {
        "search_strategies": [
            {
                "name": "ambiguous_topic",
                "intent_type": "disambiguation",
                "queries": [
                    "  auto research agents  ",
                    "auto research agents",
                    "automobile research automation",
                    "autonomous research agent systems for literature review and biomedical discovery with extremely long noisy wording",
                ],
                "preferred_sources": ["openalex"],
                "negative_terms": ["automobile", "vehicle", "factory automation"],
                "disambiguation": "auto research means autonomous research agents, not automobile research or generic automation",
            }
        ]
    }

    with _fake_llm_response(monkeypatch, llm_plan):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "auto research",
                "--run-id",
                "query-hygiene",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    search_plan = _read_yaml(tmp_path / "runs" / "query-hygiene" / "stage-03" / "search_plan.yaml")
    queries = _read_json(tmp_path / "runs" / "query-hygiene" / "stage-03" / "queries.json")
    strategy = search_plan["search_strategies"][0]
    assert set(strategy["negative_terms"]) >= {"automobile", "vehicle", "factory automation"}
    assert "not automobile research" in strategy["disambiguation"]
    assert queries["queries"].count("auto research agents") == 1
    assert "automobile research automation" not in queries["queries"]
    assert all(len(query) <= 80 for query in queries["queries"])


def test_stage03_drops_boundary_conflicting_strategies_without_short_topic_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    stage02_profile = {
        "topic_goal": "Find papers about autonomous research agents and automated scientific discovery.",
        "scope": ["autonomous research agents", "automated scientific discovery"],
        "boundary_notes": ["Primary interpretation is software systems for research workflows."],
        "disambiguation_hints": ["auto research means autonomous research, not automobile R&D"],
        "negative_terms": ["automotive", "autonomous driving", "self-driving", "vehicle AI"],
        "sub_questions": ["How do autonomous agents plan, search, and verify scientific work?"],
        "priorities": ["agentic research systems"],
        "review_hints": ["watch for automobile drift"],
    }
    stage03_plan = {
        "search_strategies": [
            {
                "name": "Autonomous_AI_Research_Agents",
                "intent_type": "core_concept",
                "queries": [
                    "autonomous research agents",
                    "AI agents for scientific discovery",
                    "agentic research workflows",
                    "automated literature synthesis agents",
                ],
                "preferred_sources": ["openalex", "arxiv"],
            },
            {
                "name": "Automotive_R_D_and_Autonomous_Driving",
                "intent_type": "boundary_conflict",
                "queries": [
                    "automotive R&D autonomous driving",
                    "self-driving vehicle AI research",
                    "vehicle AI research automation",
                ],
                "preferred_sources": ["openalex"],
            },
            {"name": "Extra_A", "queries": ["automated science benchmarks", "AI scientist systems"]},
            {"name": "Extra_B", "queries": ["research planning agents", "scientific workflow agents"]},
            {"name": "Extra_C", "queries": ["autonomous experiment design agents", "AI research copilots"]},
        ],
        "filters": {"min_year": 2022},
    }

    with _fake_stage02_and_stage03(
        monkeypatch,
        stage02_profile=stage02_profile,
        stage03_plan=stage03_plan,
    ):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "auto research",
                "--run-id",
                "stage03-quality-gate",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "stage03-quality-gate"
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    queries = _read_json(run_root / "stage-03" / "queries.json")
    intents = _read_json(run_root / "stage-03" / "search_intents.json")
    serialized = json.dumps(search_plan).casefold()

    assert "automotive_r_d" not in serialized
    assert "autonomous driving" not in " ".join(queries["queries"]).casefold()
    assert len(search_plan["search_strategies"]) >= 3
    assert len(queries["queries"]) >= 8
    assert len(queries["queries"]) > 9
    assert "boundary_conflict_strategy_dropped" in search_plan["quality_warnings"]
    assert "ambiguous_topic_query_budget_applied" not in search_plan["quality_warnings"]
    assert intents["quality_warnings"] == search_plan["quality_warnings"]
    assert all("provider_url" not in json.dumps(intent).casefold() for intent in intents["intents"])


def test_stage03_enforces_boundary_notes_before_intents_become_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    stage02_profile = {
        "topic_goal": "Find papers about research workflow software.",
        "scope": ["research workflow software"],
        "boundary_notes": ["Exclude staffing agencies and recruiting topics."],
        "disambiguation_hints": [],
        "negative_terms": [],
        "sub_questions": ["How do systems support research workflows?"],
        "priorities": [],
        "review_hints": [],
    }
    stage03_plan = {
        "search_strategies": [
            {
                "name": "Research_Workflow_Software",
                "queries": ["research workflow software"],
                "preferred_sources": ["openalex"],
            },
            {
                "name": "Research_Staffing_Agency",
                "queries": ["research staffing agency", "research recruiting topics"],
                "preferred_sources": ["openalex"],
            },
        ],
        "filters": {"min_year": 2022},
    }

    with _fake_stage02_and_stage03(
        monkeypatch,
        stage02_profile=stage02_profile,
        stage03_plan=stage03_plan,
    ):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "research workflow software",
                "--run-id",
                "stage03-boundary-notes",
                "--output-root",
                str(tmp_path / "runs"),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "stage03-boundary-notes"
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    queries = _read_json(run_root / "stage-03" / "queries.json")
    serialized = json.dumps(search_plan).casefold()

    assert "staffing_agency" not in serialized
    assert "staffing agency" not in " ".join(queries["queries"]).casefold()
    assert "recruiting topics" not in " ".join(queries["queries"]).casefold()
    assert "boundary_conflict_strategy_dropped" in search_plan["quality_warnings"]


def test_deterministic_fallback_for_auto_research_keeps_disambiguation_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "auto research",
            "--run-id",
            "auto-research-fallback",
            "--output-root",
            str(tmp_path / "runs"),
            "--plan-only",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    search_plan = _read_yaml(tmp_path / "runs" / "auto-research-fallback" / "stage-03" / "search_plan.yaml")
    queries = _read_json(tmp_path / "runs" / "auto-research-fallback" / "stage-03" / "queries.json")
    assert search_plan["query_generation"]["mode"] == "deterministic_fallback"
    assert "automotive research" in search_plan["search_strategies"][0]["negative_terms"]
    assert "autonomous research agents" in " ".join(search_plan["search_strategies"][0]["disambiguation"]).casefold()
    assert queries["queries"] == list(dict.fromkeys(queries["queries"]))
    assert "automotive" not in " ".join(queries["queries"]).casefold()
