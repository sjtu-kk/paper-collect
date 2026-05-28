from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from paper_collect import cli
from paper_collect import topic_search
from paper_collect.literature import Author, CitationResult, Paper, SearchBatchResult, VerificationReport, VerifyStatus, parse_bibtex_entries
from topic_intent_helpers import install_stage01_autoresume


@pytest.fixture(autouse=True)
def _isolate_real_workspace_openai_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def _paper(
    *,
    paper_id: str,
    title: str,
    citation_count: int,
    year: int,
    doi: str = "",
    arxiv_id: str = "",
    source: str = "openalex",
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=(Author(name="A. Author"),),
        year=year,
        abstract=f"{title} abstract",
        venue=f"{source} venue",
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        url=f"https://example.org/{paper_id}",
        source=source,
    )


def _patch_stage04_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    papers: list[Paper] | None = None,
    status: str = "completed",
    provider_statuses: list[dict] | None = None,
) -> dict:
    captured: dict = {}
    papers = papers or []
    provider_statuses = provider_statuses or [
        {
            "source_name": "openalex",
            "status": "completed",
            "returned_count": len(papers),
            "queries_attempted": 1,
            "warnings": [],
        },
        {
            "source_name": "semantic_scholar",
            "status": "completed",
            "returned_count": 0,
            "queries_attempted": 1,
            "warnings": [],
        },
        {
            "source_name": "arxiv",
            "status": "completed",
            "returned_count": 0,
            "queries_attempted": 1,
            "warnings": [],
        },
    ]

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:
        captured["queries"] = list(queries)
        captured["kwargs"] = dict(kwargs)
        return SearchBatchResult(
            queries_used=list(queries),
            papers=list(papers),
            provider_statuses=provider_statuses,
            status=status,
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)
    return captured


def _patch_stage05_verification(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: VerifyStatus = VerifyStatus.VERIFIED,
) -> dict:
    captured: dict = {}

    def fake_verify_citations(bib_text: str, **kwargs) -> VerificationReport:
        entries = parse_bibtex_entries(bib_text)
        captured["entry_count"] = len(entries)
        captured["kwargs"] = dict(kwargs)
        report = VerificationReport(total=len(entries))
        for entry in entries:
            result = CitationResult(
                cite_key=entry["key"],
                title=entry.get("title", ""),
                status=status,
                confidence=1.0,
                method="test_verifier",
                details="verified in test",
            )
            report.results.append(result)
            if status == VerifyStatus.VERIFIED:
                report.verified += 1
            elif status == VerifyStatus.SUSPICIOUS:
                report.suspicious += 1
            elif status == VerifyStatus.HALLUCINATED:
                report.hallucinated += 1
            else:
                report.skipped += 1
        return report

    monkeypatch.setattr(topic_search, "verify_citations", fake_verify_citations, raising=False)
    return captured


def test_topic_search_cli_writes_stage03_scaffold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    _patch_stage04_search(monkeypatch, papers=[])
    _patch_stage05_verification(monkeypatch)
    output_root = tmp_path / "runtime_data" / "topic_search_runs"

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "scaffold-smoke",
            "--output-root",
            str(output_root),
            "--max-results-per-query",
            "7",
            "--year-min",
            "2021",
            "--inter-query-delay",
            "0.1",
            "--inter-verify-delay",
            "0.2",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "run_meta:" in output
    assert "stage-03/search_plan.yaml" in output.replace("\\", "/")

    run_root = output_root / "scaffold-smoke"
    run_meta = _read_json(run_root / "run_meta.json")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    sources = _read_json(run_root / "stage-03" / "sources.json")
    queries = _read_json(run_root / "stage-03" / "queries.json")

    assert run_meta["artifact_version"] == "topic_search_run.v1"
    assert run_meta["run_id"] == "scaffold-smoke"
    assert run_meta["request"] == {
        "topic": "AI agents for biomedical research",
        "description": "",
        "max_results_per_query": 7,
        "year_min": 2021,
        "s2_api_key_provided": False,
        "inter_query_delay": 0.1,
        "inter_verify_delay": 0.2,
    }
    assert run_meta["stage_statuses"]["stage-03"] == "complete"

    stage03_refs = run_meta["stage_artifacts"]["stage-03"]
    assert stage03_refs["search_plan_yaml"] == "stage-03/search_plan.yaml"
    assert stage03_refs["sources_json"] == "stage-03/sources.json"
    assert stage03_refs["queries_json"] == "stage-03/queries.json"

    assert search_plan["artifact_version"] == "topic_search_search_plan.v1"
    assert search_plan["stage"] == "stage-03"
    assert search_plan["topic"] == "AI agents for biomedical research"
    assert search_plan["query_generation"]["reference_plan_labels"] == [
        "openreview",
        "google_scholar",
    ]
    assert search_plan["search_strategies"][0]["sources"] == [
        "openalex",
        "semantic_scholar",
        "arxiv",
        "crossref",
        "dblp",
        "doaj",
        "pubmed",
    ]

    active_sources = [source for source in sources["sources"] if source["status"] == "active"]
    reference_labels = [source for source in sources["sources"] if source["status"] == "reference_only"]
    assert [source["source_name"] for source in active_sources] == [
        "openalex",
        "semantic_scholar",
        "arxiv",
        "crossref",
        "dblp",
        "doaj",
        "pubmed",
    ]
    assert [source["source_name"] for source in reference_labels] == [
        "openreview",
        "google_scholar",
    ]
    assert queries["topic"] == "AI agents for biomedical research"
    assert queries["year_min"] == 2021
    assert queries["max_results_per_query"] == 7
    assert len(queries["queries"]) >= 3


def test_topic_search_stage04_writes_candidates_references_and_search_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _patch_stage04_search(
        monkeypatch,
        papers=[
            _paper(
                paper_id="p-1",
                title="Alpha Paper",
                citation_count=12,
                year=2025,
                doi="10.1000/alpha",
                source="openalex",
            ),
            _paper(
                paper_id="p-2",
                title="Beta Paper",
                citation_count=7,
                year=2024,
                arxiv_id="2401.00001",
                source="arxiv",
            ),
        ],
    )
    _patch_stage05_verification(monkeypatch)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="stage04-run",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=6,
        year_min=2021,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        allow_deterministic_fallback=True,
    )

    run_root = result.run_root
    candidates_path = run_root / "stage-04" / "candidates.jsonl"
    references_path = run_root / "stage-04" / "references.bib"
    search_meta_path = run_root / "stage-04" / "search_meta.json"
    run_meta = _read_json(result.run_meta_path)

    assert run_meta["stage_statuses"]["stage-04"] == "complete"
    assert candidates_path.is_file()
    assert references_path.is_file()
    assert search_meta_path.is_file()

    candidate_lines = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [candidate["title"] for candidate in candidate_lines] == ["Alpha Paper", "Beta Paper"]
    assert candidate_lines[0]["citation_count"] == 12
    assert candidate_lines[1]["arxiv_id"] == "2401.00001"
    assert [candidate["source"] for candidate in candidate_lines] == ["openalex", "arxiv"]
    assert references_path.read_text(encoding="utf-8").strip().startswith("@article{")

    search_meta = _read_json(search_meta_path)
    assert search_meta["status"] == "completed"
    assert search_meta["real_search"] is True
    assert search_meta["queries_used"] == captured["queries"]
    assert search_meta["year_min"] == 2021
    assert search_meta["total_candidates"] == 2
    assert search_meta["bibtex_entries"] == 2
    assert search_meta["provider_statuses"][0]["source_name"] == "openalex"
    assert captured["kwargs"]["limit_per_query"] == 6
    assert captured["kwargs"]["year_min"] == 2021
    assert captured["kwargs"]["sources"] == ("openalex", "semantic_scholar", "arxiv", "crossref", "dblp", "doaj", "pubmed")
    assert not (run_root / "stage-04" / "provider_raw").exists()
    assert not (run_root / "stage-04" / "verification_meta.json").exists()


def test_topic_search_stage04_records_provider_executions_and_minimal_candidate_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(
        monkeypatch,
        papers=[
            _paper(
                paper_id="p-1",
                title="Alpha Paper",
                citation_count=12,
                year=2025,
                doi="10.1000/alpha",
                source="openalex",
            ),
        ],
        provider_statuses=[
            {
                "source_name": "openalex",
                "status": "completed",
                "returned_count": 1,
                "queries_attempted": 1,
                "warnings": [],
            },
            {
                "source_name": "semantic_scholar",
                "status": "rate_limited",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": ["429"],
            },
            {
                "source_name": "arxiv",
                "status": "network_failed",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": ["offline"],
            },
            {
                "source_name": "crossref",
                "status": "provider_error",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": ["Crossref unavailable"],
            },
            {
                "source_name": "dblp",
                "status": "provider_error",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": ["DBLP unavailable"],
            },
            {
                "source_name": "doaj",
                "status": "provider_error",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": ["DOAJ unavailable"],
            },
            {
                "source_name": "pubmed",
                "status": "provider_error",
                "returned_count": 0,
                "queries_attempted": 1,
                "warnings": ["PubMed unavailable"],
            },
        ],
    )
    _patch_stage05_verification(monkeypatch)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="provider-executions-run",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=6,
        year_min=2021,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate_lines = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert search_meta["query_source"] == "search_intents"
    assert search_meta["search_intents_contract_ref"] == "stage-03/search_intents.json"
    assert len(search_meta["provider_executions"]) == 7
    assert search_meta["provider_execution_tasks"]
    assert search_meta["provider_executions"][0]["provider_execution_id"] == "provider_execution_openalex"
    assert search_meta["provider_executions"][0]["provider_id"] == "openalex"
    assert search_meta["provider_executions"][0]["provider_name"] == "OpenAlex"
    assert search_meta["provider_executions"][0]["intended_surface"]
    assert search_meta["provider_executions"][0]["compiled_query_summary"]["query_source"] == "search_intents"
    assert search_meta["provider_executions"][0]["compiled_query_summary"]["query_count"] >= 1
    assert search_meta["provider_executions"][0]["applied_filters"] == {"year_min": 2021}
    assert search_meta["provider_executions"][0]["status"] == "completed"
    assert search_meta["provider_executions"][1]["status"] == "rate_limited"
    assert search_meta["provider_executions"][2]["status"] == "network"
    assert search_meta["provider_executions"][3]["status"] == "blocked"
    assert search_meta["provider_executions"][4]["status"] == "blocked"
    assert search_meta["provider_executions"][5]["status"] == "blocked"
    assert search_meta["provider_executions"][6]["status"] == "blocked"
    assert candidate_lines[0]["match_provenance"]["provider_execution_id"] in {
        task["provider_execution_id"] for task in search_meta["provider_execution_tasks"]
    }
    assert candidate_lines[0]["match_provenance"]["provider_result_index"] == 1
    assert set(candidate_lines[0]["match_provenance"]).issubset(
        {
            "provider_execution_id",
            "provider_result_index",
            "provider_native_identifiers",
        }
    )
    assert "matched_field" not in candidate_lines[0]


def test_stage04_records_active_provider_surface_audit_and_executed_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(
        monkeypatch,
        papers=[
            _paper(
                paper_id="p-surface",
                title="Surface Audit Paper",
                citation_count=5,
                year=2025,
                doi="10.1000/surface",
                source="openalex",
            )
        ],
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
        run_id="provider-surface-audit",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    audit = search_meta["provider_surface_audit"]
    providers = {entry["provider_id"]: entry for entry in audit["providers"]}

    assert audit["artifact_version"] == "topic_search_provider_surface_audit.v1"
    assert set(providers) == {"openalex", "semantic_scholar", "arxiv", "crossref", "dblp", "doaj", "pubmed"}
    assert providers["openalex"]["supported_search_surfaces"] == ["works.search", "title.search", "primary_location.source"]
    assert providers["openalex"]["executed_surface"] == "works.search"
    assert providers["semantic_scholar"]["surface_gap_state"] == "rate_limited_provider_local_backoff"
    assert providers["arxiv"]["surface_gap_state"] == "rate_limited_provider_local_backoff"
    assert providers["dblp"]["coverage_judgment"] == "needs_provider_local_compaction"
    assert providers["dblp"]["surface_gap_state"] == "provider_local_compaction_applied"
    assert providers["pubmed"]["supported_search_surfaces"] == ["esearch", "esummary"]
    assert providers["pubmed"]["executed_surface"] == "esearch+esummary"
    assert all("executed_surface" in execution for execution in search_meta["provider_executions"])
    assert all("executed_surface" in task for task in search_meta["provider_execution_tasks"])
    assert "matched_field" not in candidate
    assert set(candidate["match_provenance"]).issubset(
        {
            "provider_execution_id",
            "provider_result_index",
            "provider_native_identifiers",
        }
    )


def test_stage04_retains_task_level_match_and_dedup_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    winner = _paper(
        paper_id="s2-winner",
        title="Autonomous Research Agents",
        citation_count=20,
        year=2025,
        doi="10.1000/agents",
        source="semantic_scholar",
    )
    loser = _paper(
        paper_id="oa-loser",
        title="Autonomous Research Agents",
        citation_count=3,
        year=2024,
        doi="10.1000/agents",
        source="openalex",
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        first_query = queries[0]
        second_query = queries[1]
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[winner],
            provider_statuses=[
                {
                    "source_name": "openalex",
                    "status": "completed",
                    "returned_count": 1,
                    "queries_attempted": 2,
                    "warnings": [],
                },
                {
                    "source_name": "semantic_scholar",
                    "status": "completed",
                    "returned_count": 2,
                    "queries_attempted": 2,
                    "warnings": [],
                },
                {
                    "source_name": "arxiv",
                    "status": "completed",
                    "returned_count": 0,
                    "queries_attempted": 2,
                    "warnings": [],
                },
            ],
            provider_result_rows=[
                {
                    "source_name": "semantic_scholar",
                    "query_text": first_query,
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [winner.to_dict()],
                },
                {
                    "source_name": "semantic_scholar",
                    "query_text": second_query,
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [winner.to_dict()],
                },
                {
                    "source_name": "openalex",
                    "query_text": first_query,
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [loser.to_dict()],
                },
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="task-ledger-run",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    run_meta = _read_json(result.run_meta_path)
    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger_rows = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "match_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    task_by_id = {
        task["provider_execution_id"]: task
        for task in search_meta["provider_execution_tasks"]
    }
    provenance = candidate["match_provenance"]
    assert run_meta["stage_artifacts"]["stage-04"]["match_ledger_jsonl"] == "stage-04/match_ledger.jsonl"
    assert provenance["provider_execution_id"] in task_by_id
    assert task_by_id[provenance["provider_execution_id"]]["provider_id"] == "semantic_scholar"
    assert task_by_id[provenance["provider_execution_id"]]["intent_id"]
    assert task_by_id[provenance["provider_execution_id"]]["query_text"]
    assert provenance["provider_result_index"] == 1
    assert {row["hit_type"] for row in ledger_rows} >= {"primary", "additional_hit", "dedup_loser"}
    assert any(row["provider_id"] == "openalex" and row["hit_type"] == "dedup_loser" for row in ledger_rows)
    assert all("request_params" not in row for row in ledger_rows)


def test_stage04_surfaces_openalex_source_authority_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_metadata = {
        "evidence_kind": "openalex_source_authority",
        "source_id": "https://openalex.org/S4210172161",
        "display_name": "Proceedings of Machine Learning Research",
        "issn_l": "2640-3498",
        "issn": ["2640-3498"],
        "type": "conference",
        "host_organization": "https://openalex.org/P4310320990",
    }
    paper = Paper(
        paper_id="https://openalex.org/W-source",
        title="OpenAlex Source Authority Evidence",
        authors=(Author(name="A. Author"),),
        year=2026,
        abstract="source evidence abstract",
        venue="Proceedings of Machine Learning Research",
        citation_count=11,
        doi="10.1000/source-authority",
        url="https://openalex.org/W-source",
        source="openalex",
        source_metadata=source_metadata,
    )

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        first_query = queries[0]
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[paper],
            provider_statuses=[
                {
                    "source_name": "openalex",
                    "status": "completed",
                    "returned_count": 1,
                    "queries_attempted": 1,
                    "warnings": [],
                },
                {
                    "source_name": "semantic_scholar",
                    "status": "completed",
                    "returned_count": 0,
                    "queries_attempted": 1,
                    "warnings": [],
                },
                {
                    "source_name": "arxiv",
                    "status": "completed",
                    "returned_count": 0,
                    "queries_attempted": 1,
                    "warnings": [],
                },
            ],
            provider_result_rows=[
                {
                    "source_name": "openalex",
                    "query_text": first_query,
                    "status": "completed",
                    "returned_count": 1,
                    "warnings": [],
                    "papers": [paper.to_dict()],
                },
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="openalex-source-authority",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger_rows = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "match_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    openalex_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "openalex")
    assert candidate["source_metadata"] == source_metadata
    assert openalex_task["source_authority_evidence"]["normalized_count"] == 1
    assert openalex_task["source_authority_evidence"]["sources"][0] == {
        "source_id": "https://openalex.org/S4210172161",
        "display_name": "Proceedings of Machine Learning Research",
        "issn_l": "2640-3498",
        "type": "conference",
    }
    assert ledger_rows[0]["source_authority_evidence"] == {
        "source_id": "https://openalex.org/S4210172161",
        "display_name": "Proceedings of Machine Learning Research",
        "issn_l": "2640-3498",
        "type": "conference",
    }
    assert "source_authority_evidence" not in candidate["match_provenance"]


def test_stage04_surfaces_crossref_authority_evidence(
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
        title="Crossref Authority Agents",
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
        run_id="crossref-authority",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger_rows = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "match_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    crossref_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "crossref")
    assert candidate["source_metadata"] == authority_metadata
    assert candidate["match_provenance"] == {
        "provider_execution_id": crossref_task["provider_execution_id"],
        "provider_result_index": 1,
        "provider_native_identifiers": {"doi": "10.1145/1234567.8901234"},
    }
    assert crossref_task["bibliographic_authority_evidence"] == {
        "evidence_kind": "crossref_bibliographic_authority",
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
    assert ledger_rows[0]["bibliographic_authority_evidence"] == {
        "crossref_id": "10.1145/1234567.8901234",
        "container_title": "Proceedings of the ACM Conference on Research Agents",
        "issn": ["1234-5678"],
        "publisher": "Association for Computing Machinery",
        "type": "proceedings-article",
    }
    assert "bibliographic_authority_evidence" not in candidate["match_provenance"]


def test_stage04_surfaces_dblp_cs_bibliography_evidence(
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
        run_id="dblp-cs-bibliography",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger_rows = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "match_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    dblp_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "dblp")
    assert candidate["source_metadata"] == cs_metadata
    assert candidate["abstract"] == ""
    assert candidate["match_provenance"] == {
        "provider_execution_id": dblp_task["provider_execution_id"],
        "provider_result_index": 1,
        "provider_native_identifiers": {"doi": "10.48550/arXiv.1710.10903"},
    }
    assert dblp_task["cs_bibliography_evidence"] == {
        "evidence_kind": "dblp_cs_bibliographic",
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
    assert ledger_rows[0]["cs_bibliography_evidence"] == {
        "dblp_key": "conf/iclr/VelickovicCCRLB18",
        "venue": "ICLR",
        "publication_type": "Conference and Workshop Papers",
        "ee": "https://doi.org/10.48550/arXiv.1710.10903",
    }
    assert "cs_bibliography_evidence" not in candidate["match_provenance"]


def test_stage04_records_dblp_compacted_query_and_no_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    long_query = "graph attention networks for interpretable neural message passing with irrelevant provider prose and survey framing"

    def fake_search_papers_multi_query(queries: list[str], **kwargs) -> SearchBatchResult:  # noqa: ANN001
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[],
            provider_statuses=[
                {"source_name": "openalex", "status": "completed", "returned_count": 1, "queries_attempted": 1, "warnings": []},
                {"source_name": "semantic_scholar", "status": "rate_limited", "returned_count": 0, "queries_attempted": 1, "warnings": ["Semantic Scholar rate limited"], "rate_control": {"provider_local": True}},
                {"source_name": "arxiv", "status": "rate_limited", "returned_count": 0, "queries_attempted": 1, "warnings": ["arXiv rate limited"], "rate_control": {"provider_local": True}},
                {"source_name": "crossref", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "dblp", "status": "completed_no_results", "returned_count": 0, "queries_attempted": 1, "warnings": [], "compiled_query_summary": {"original_query_text": long_query, "provider_query_text": "graph attention networks interpretable neural message passing", "rewrite_reason": "dblp_keyword_compaction"}},
                {"source_name": "doaj", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
                {"source_name": "pubmed", "status": "completed", "returned_count": 0, "queries_attempted": 1, "warnings": []},
            ],
            provider_result_rows=[
                {
                    "source_name": "dblp",
                    "query_text": long_query,
                    "status": "completed_no_results",
                    "returned_count": 0,
                    "warnings": [],
                    "compiled_query_summary": {
                        "original_query_text": long_query,
                        "provider_query_text": "graph attention networks interpretable neural message passing",
                        "rewrite_reason": "dblp_keyword_compaction",
                    },
                    "papers": [],
                }
            ],
            status="completed",
            real_search=True,
        )

    monkeypatch.setattr(topic_search, "search_papers_multi_query", fake_search_papers_multi_query, raising=False)

    result = topic_search.run_topic_search(
        topic="graph attention networks",
        run_id="dblp-no-results-polish",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_intents = _read_json(result.run_root / "stage-03" / "search_intents.json")
    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    dblp_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "dblp")
    s2_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "semantic_scholar")
    arxiv_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "arxiv")

    assert all(intent["query_text"] != "graph attention networks interpretable neural message passing" for intent in search_intents["intents"])
    assert dblp_task["status"] == "completed_no_results"
    assert dblp_task["compiled_query_summary"] == {
        "query_source": "search_intents",
        "query_text": long_query,
        "max_results_per_query": 5,
        "original_query_text": long_query,
        "provider_query_text": "graph attention networks interpretable neural message passing",
        "rewrite_reason": "dblp_keyword_compaction",
    }
    assert s2_task["rate_control"] == {"provider_local": True}
    assert arxiv_task["rate_control"] == {"provider_local": True}


def test_stage04_surfaces_doaj_oa_metadata_evidence(
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
        run_id="doaj-oa-metadata",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger_rows = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "match_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    doaj_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "doaj")
    assert candidate["source_metadata"] == oa_metadata
    assert candidate["match_provenance"] == {
        "provider_execution_id": doaj_task["provider_execution_id"],
        "provider_result_index": 1,
        "provider_native_identifiers": {"doi": "10.1234/doaj.1"},
    }
    assert doaj_task["oa_metadata_evidence"] == {
        "evidence_kind": "doaj_oa_metadata",
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
    assert ledger_rows[0]["oa_metadata_evidence"]["doaj_id"] == "doaj-article-1"
    assert "oa_metadata_evidence" not in candidate["match_provenance"]


def test_stage04_surfaces_pubmed_biomedical_corpus_evidence(
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
        run_id="pubmed-biomedical-corpus",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        max_results_per_query=5,
        inter_query_delay=0.0,
        inter_verify_delay=0.0,
        verification_enabled=False,
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidate = json.loads((result.run_root / "stage-04" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ledger_rows = [
        json.loads(line)
        for line in (result.run_root / "stage-04" / "match_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    pubmed_task = next(task for task in search_meta["provider_execution_tasks"] if task["provider_id"] == "pubmed")
    assert candidate["source_metadata"] == biomedical_metadata
    assert candidate["match_provenance"] == {
        "provider_execution_id": pubmed_task["provider_execution_id"],
        "provider_result_index": 1,
        "provider_native_identifiers": {"doi": "10.1109/JBHI.2026.3696620"},
    }
    assert pubmed_task["biomedical_corpus_evidence"] == {
        "evidence_kind": "pubmed_biomedical_corpus",
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
    assert ledger_rows[0]["biomedical_corpus_evidence"]["pmid"] == "42184189"
    assert "biomedical_corpus_evidence" not in candidate["match_provenance"]


def test_topic_search_stage04_marks_blocked_or_failed_without_placeholder_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(
        monkeypatch,
        status="blocked_or_failed",
        provider_statuses=[
            {"source_name": "openalex", "status": "provider_error", "returned_count": 0, "queries_attempted": 1, "warnings": ["boom"]},
            {"source_name": "semantic_scholar", "status": "rate_limited", "returned_count": 0, "queries_attempted": 1, "warnings": ["429"]},
            {"source_name": "arxiv", "status": "network_failed", "returned_count": 0, "queries_attempted": 1, "warnings": ["offline"]},
        ],
    )
    _patch_stage05_verification(monkeypatch)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="blocked-stage04-run",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        allow_deterministic_fallback=True,
    )

    search_meta = _read_json(result.run_root / "stage-04" / "search_meta.json")
    candidates_path = result.run_root / "stage-04" / "candidates.jsonl"
    references_path = result.run_root / "stage-04" / "references.bib"

    assert search_meta["status"] == "blocked_or_failed"
    assert search_meta["total_candidates"] == 0
    assert candidates_path.read_text(encoding="utf-8").strip() == ""
    assert references_path.read_text(encoding="utf-8").strip() == ""
    assert _read_json(result.run_meta_path)["stage_statuses"]["stage-04"] == "blocked_or_failed"


def test_topic_search_stage05_verifies_every_stage04_bibtex_entry_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(
        monkeypatch,
        papers=[
            _paper(paper_id="p-1", title="Alpha Paper", citation_count=12, year=2025, doi="10.1000/alpha"),
            _paper(paper_id="p-2", title="Beta Paper", citation_count=7, year=2024, arxiv_id="2401.00001", source="arxiv"),
        ],
    )
    captured = _patch_stage05_verification(monkeypatch, status=VerifyStatus.VERIFIED)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="stage05-run",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        inter_query_delay=0.0,
        inter_verify_delay=0.25,
        allow_deterministic_fallback=True,
    )

    report_path = result.run_root / "stage-05" / "verification_report.json"
    verified_bib_path = result.run_root / "stage-05" / "references_verified.bib"
    run_meta = _read_json(result.run_meta_path)

    assert run_meta["stage_statuses"]["stage-05"] == "complete"
    assert captured["entry_count"] == 2
    assert captured["kwargs"]["inter_verify_delay"] == 0.25
    assert report_path.is_file()
    assert verified_bib_path.is_file()
    report = _read_json(report_path)
    assert report["artifact_version"] == "topic_search_verification_report.v1"
    assert report["summary"]["total"] == 2
    assert report["summary"]["verified"] == 2
    assert [item["method"] for item in report["results"]] == ["test_verifier", "test_verifier"]
    assert "Alpha Paper" in verified_bib_path.read_text(encoding="utf-8")
    assert "Beta Paper" in verified_bib_path.read_text(encoding="utf-8")


def test_topic_search_stage05_handles_empty_bibliography_without_fabricating_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(monkeypatch, status="blocked_or_failed", provider_statuses=[
        {"source_name": "openalex", "status": "provider_error", "returned_count": 0, "queries_attempted": 1, "warnings": ["boom"]},
        {"source_name": "semantic_scholar", "status": "rate_limited", "returned_count": 0, "queries_attempted": 1, "warnings": ["429"]},
        {"source_name": "arxiv", "status": "network_failed", "returned_count": 0, "queries_attempted": 1, "warnings": ["offline"]},
    ])
    _patch_stage05_verification(monkeypatch)

    result = topic_search.run_topic_search(
        topic="AI agents for biomedical research",
        run_id="empty-stage05-run",
        output_root=tmp_path / "runtime_data" / "topic_search_runs",
        allow_deterministic_fallback=True,
    )

    report = _read_json(result.run_root / "stage-05" / "verification_report.json")
    assert report["status"] == "blocked_or_failed"
    assert report["summary"]["total"] == 0
    assert report["results"] == []
    assert (result.run_root / "stage-05" / "references_verified.bib").read_text(encoding="utf-8") == ""
    assert _read_json(result.run_meta_path)["stage_statuses"]["stage-05"] == "blocked_or_failed"


def test_topic_search_cli_runs_closed_loop_with_env_api_key_and_exact_artifact_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_search = _patch_stage04_search(
        monkeypatch,
        papers=[_paper(paper_id="p-1", title="Alpha Paper", citation_count=12, year=2025, doi="10.1000/alpha")],
    )
    captured_verify = _patch_stage05_verification(monkeypatch)
    monkeypatch.setenv("S2_API_KEY", "env-key")
    output_root = tmp_path / "runtime_data" / "topic_search_runs"

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "closed-loop",
            "--output-root",
            str(output_root),
            "--max-results-per-query",
            "5",
            "--year-min",
            "2022",
            "--inter-query-delay",
            "0",
            "--inter-verify-delay",
            "0",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    run_root = output_root / "closed-loop"
    all_files = sorted(str(path.relative_to(run_root)).replace("\\", "/") for path in run_root.rglob("*") if path.is_file())
    assert all_files == [
        "run_meta.json",
        "stage-01/topic_intent.md",
        "stage-01/topic_intent_draft.md",
        "stage-02/topic_profile.md",
        "stage-03/queries.json",
        "stage-03/search_intents.json",
        "stage-03/search_plan.yaml",
        "stage-03/sources.json",
        "stage-04/candidates.jsonl",
        "stage-04/match_ledger.jsonl",
        "stage-04/references.bib",
        "stage-04/search_meta.json",
        "stage-05/references_verified.bib",
        "stage-05/verification_report.json",
    ]
    run_meta = _read_json(run_root / "run_meta.json")
    assert run_meta["stage_statuses"]["stage-01"] == "confirmed"
    assert run_meta["stage_statuses"]["stage-02"] == "complete"
    assert run_meta["stage_statuses"]["stage-03"] == "complete"
    assert run_meta["stage_statuses"]["stage-04"] == "complete"
    assert run_meta["stage_statuses"]["stage-05"] == "complete"
    assert run_meta["cache"] == {"enabled": False, "can_substitute_provider_results": False}
    assert run_meta["request"]["s2_api_key_provided"] is True
    assert captured_search["kwargs"]["s2_api_key"] == "env-key"
    assert captured_verify["kwargs"]["s2_api_key"] == "env-key"


def test_topic_search_cli_search_only_disables_verification_and_review_renders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_search = _patch_stage04_search(
        monkeypatch,
        papers=[_paper(paper_id="p-1", title="Alpha Paper", citation_count=12, year=2025, doi="10.1000/alpha")],
    )

    def fail_if_verified(*args, **kwargs):  # noqa: ANN001, ANN202
        raise AssertionError("Stage 05 verification should be disabled for this run")

    monkeypatch.setattr(topic_search, "verify_citations", fail_if_verified, raising=False)
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
            "--inter-query-delay",
            "0",
            "--disable-verification",
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
    assert run_meta["verification"] == {"enabled": False, "status": "disabled"}
    assert not (run_root / "stage-05" / "verification_report.json").exists()
    assert captured_search["kwargs"]["limit_per_query"] == 5

    renderer = Path(__file__).parents[1] / "scripts" / "render_topic_search_review.py"
    completed = subprocess.run(
        [sys.executable, str(renderer), str(run_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    review_html = run_root / "review.html"
    assert review_html.is_file()
    assert '"status":"disabled"' in review_html.read_text(encoding="utf-8")


def test_topic_search_cli_keeps_max_results_per_query_default_40(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_search = _patch_stage04_search(monkeypatch, papers=[])
    output_root = tmp_path / "runtime_data" / "topic_search_runs"

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "AI agents for biomedical research",
            "--run-id",
            "default-limit",
            "--output-root",
            str(output_root),
            "--inter-query-delay",
            "0",
            "--disable-verification",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    assert captured_search["kwargs"]["limit_per_query"] == 40


def test_topic_search_cli_invalid_topic_returns_nonzero(capsys) -> None:
    exit_code = cli.main(["topic-search", "--topic", "   "])

    assert exit_code == 2
    assert "topic must not be empty" in capsys.readouterr().err


def test_topic_search_cli_uses_workspace_default_output_root_when_run_id_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_stage04_search(monkeypatch, papers=[])
    _patch_stage05_verification(monkeypatch)
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["topic-search", "--topic", "biomedical ai agent", "--allow-deterministic-fallback"])

    assert exit_code == 0
    output_root = tmp_path / "runtime_data" / "topic_search_runs"
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_meta = _read_json(run_dirs[0] / "run_meta.json")
    assert run_meta["request"]["topic"] == "biomedical ai agent"
    assert run_meta["stage_statuses"]["stage-03"] == "complete"


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "--request-class", "exact_title", "--query", "Attention Is All You Need"],
        ["benchmark-search-diagnostics"],
        ["compare-reference-baselines"],
        ["evaluate-search-foundation"],
        ["topic-search", "--topic", "biomedical ai agent", "--request-class", "exact_title"],
        ["topic-search", "--topic", "biomedical ai agent", "--source-url", "https://example.org/paper"],
    ],
)
def test_topic_search_cli_rejects_unsupported_inputs(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(argv)
