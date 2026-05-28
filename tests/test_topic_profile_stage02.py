from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from paper_collect import cli
from topic_intent_helpers import write_confirmed_topic_intent


@pytest.fixture(autouse=True)
def _isolate_real_workspace_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_workspace = tmp_path / "env-isolated-paper-collect"
    fake_package_dir = fake_workspace / "src" / "paper_collect"
    fake_package_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_package_dir / "cli.py"))
    monkeypatch.chdir(tmp_path)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@contextmanager
def _fake_llm_response(monkeypatch: pytest.MonkeyPatch, payload: dict | str):  # noqa: ANN001
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            content = payload if isinstance(payload, str) else json.dumps(payload)
            response = {"choices": [{"message": {"content": content}}]}
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        return FakeResponse()

    from paper_collect import topic_search

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fake_urlopen)
    yield


@contextmanager
def _fake_stage02_profile_and_stage03_fallback(
    monkeypatch: pytest.MonkeyPatch,
    profile: dict,
):  # noqa: ANN001
    class FakeResponse:
        def __init__(self, content: dict | str):
            self._content = content

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            content = self._content if isinstance(self._content, str) else json.dumps(self._content)
            response = {"choices": [{"message": {"content": content}}]}
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        body = json.loads(request.data.decode("utf-8"))
        prompt = body["messages"][1]["content"]
        if "Stage 02-lite topic profile" in prompt:
            return FakeResponse({"topic_profile": profile})
        if "Stage 03 literature-search query plan" in prompt:
            return FakeResponse({})
        raise AssertionError(f"unexpected prompt: {prompt}")

    from paper_collect import topic_search

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fake_urlopen)
    yield


def test_stage02_topic_profile_is_written_before_stage03_and_consumed_by_plan(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"
    run_id = "stage02-fallback"
    write_confirmed_topic_intent(output_root / run_id, topic="auto research")

    exit_code = cli.main(
        [
            "topic-search",
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--resume",
            "--plan-only",
            "--allow-deterministic-fallback",
        ]
    )

    assert exit_code == 0
    run_root = tmp_path / "runs" / "stage02-fallback"
    topic_profile_path = run_root / "stage-02" / "topic_profile.md"
    search_plan_path = run_root / "stage-03" / "search_plan.yaml"
    assert topic_profile_path.is_file()

    profile = _read_yaml(topic_profile_path)
    assert profile["artifact_version"] == "topic_search_topic_profile.v1"
    assert profile["stage"] == "stage-02"
    assert profile["topic"] == "auto research"
    assert profile["generation"]["mode"] == "deterministic_fallback"
    assert "automotive research" in profile["negative_terms"]
    assert "autonomous research agents" in " ".join(profile["disambiguation_hints"]).casefold()

    run_meta = _read_json(run_root / "run_meta.json")
    assert run_meta["stage_statuses"]["stage-02"] == "complete"
    assert run_meta["stage_artifacts"]["stage-02"]["topic_profile_md"] == "stage-02/topic_profile.md"
    assert run_meta["topic_profile_generation"] == {
        "mode": "deterministic_fallback",
        "fallback_reason": "missing_openai_env",
    }

    search_plan = _read_yaml(search_plan_path)
    assert search_plan["topic_profile_ref"] == "../stage-02/topic_profile.md"
    assert "autonomously search and synthesize research literature" in search_plan["topic_profile"]["topic_goal"]
    assert set(search_plan["topic_profile"]["negative_terms"]) >= {
        "automotive research",
        "self-driving vehicles",
    }


def test_stage02_llm_topic_profile_uses_same_shape_and_preserves_review_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    llm_profile = {
        "topic_goal": "Find papers about autonomous AI research agents for biomedical work",
        "scope": ["software agents", "biomedical literature discovery"],
        "boundary_notes": ["exclude human staffing agencies"],
        "disambiguation_hints": ["agent means software agent"],
        "negative_terms": ["real estate", "staffing agency"],
        "sub_questions": [
            "How are autonomous AI agents used to search, synthesize, and verify biomedical literature?"
        ],
        "priorities": ["recent survey papers"],
        "review_hints": ["check whether systems verify citations"],
        "provider_selection": ["openalex"],
        "bibtex": "@article{fake}",
        "candidates": [{"title": "Fake Paper"}],
    }
    output_root = tmp_path / "runs"
    run_id = "stage02-llm"
    write_confirmed_topic_intent(
        output_root / run_id,
        topic="AI agents for biomedical research",
        intended_meaning=["software agents for biomedical research"],
        research_purpose=["Find agent systems for biomedical literature discovery"],
        included_concepts=["software agents", "biomedical literature discovery"],
        excluded_meanings=["real estate agents", "staffing agency"],
    )

    with _fake_llm_response(monkeypatch, {"topic_profile": llm_profile}):
        exit_code = cli.main(
            [
                "topic-search",
                "--run-id",
                run_id,
                "--output-root",
                str(output_root),
                "--resume",
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    profile = _read_yaml(tmp_path / "runs" / "stage02-llm" / "stage-02" / "topic_profile.md")
    assert profile["generation"]["mode"] == "llm"
    assert profile["topic_goal"] == "Find papers about autonomous AI research agents for biomedical work"
    assert profile["scope"] == ["software agents", "biomedical literature discovery"]
    assert "exclude human staffing agencies" in profile["boundary_notes"]
    assert "software agents for biomedical research" in profile["disambiguation_hints"]
    assert "agent means software agent" in profile["disambiguation_hints"]
    assert set(profile["negative_terms"]) >= {"real estate agents", "staffing agency", "real estate"}
    assert profile["sub_questions"] == [
        "How are autonomous AI agents used to search, synthesize, and verify biomedical literature?"
    ]
    assert profile["priorities"] == ["recent survey papers"]
    assert profile["review_hints"] == ["check whether systems verify citations"]
    run_meta = _read_json(tmp_path / "runs" / "stage02-llm" / "run_meta.json")
    assert run_meta["topic_profile_generation"] == {
        "mode": "llm",
        "fallback_reason": "",
    }
    assert {
        "artifact_version",
        "stage",
        "topic",
        "generated_at",
        "generation",
        "topic_goal",
        "scope",
        "boundary_notes",
        "disambiguation_hints",
        "negative_terms",
        "sub_questions",
        "priorities",
        "review_hints",
    }.issubset(profile)
    profile_text = json.dumps(profile).casefold()
    assert "provider_selection" not in profile_text
    assert "bibtex" not in profile_text
    assert "fake paper" not in profile_text


def test_stage02_llm_profile_records_quality_warnings_instead_of_silent_sparse_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    dangling_goal = (
        "Find papers about autonomous research agents, automated science, and AI systems "
        "that plan and execute research workflows while accommodating the secondary interpretation of a"
    )
    sparse_profile = {
        "topic_goal": dangling_goal,
        "scope": [],
        "boundary_notes": [],
        "disambiguation_hints": [],
        "negative_terms": [],
        "sub_questions": ["How does automotive R&D use autonomous driving research?"],
        "priorities": [],
        "review_hints": [],
    }
    output_root = tmp_path / "runs"
    run_id = "stage02-quality-warnings"
    write_confirmed_topic_intent(output_root / run_id, topic="auto research")

    with _fake_stage02_profile_and_stage03_fallback(monkeypatch, sparse_profile):
        exit_code = cli.main(
            [
                "topic-search",
                "--run-id",
                run_id,
                "--output-root",
                str(output_root),
                "--resume",
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    profile = _read_yaml(tmp_path / "runs" / "stage02-quality-warnings" / "stage-02" / "topic_profile.md")
    assert profile["generation"]["mode"] == "llm"
    assert not profile["topic_goal"].endswith(" of a")
    assert "automotive R&D" not in " ".join(profile["sub_questions"])
    assert profile["primary_interpretation"]
    assert profile["excluded_interpretations"]
    assert "automotive" in " ".join(profile["negative_terms"]).casefold()
    assert {
        "dangling_topic_goal_repaired",
        "excluded_interpretation_removed_from_sub_questions",
    }.issubset(set(profile["quality_warnings"]))


def test_stage02_repairs_primary_from_confirmed_intent_when_llm_selects_excluded_meaning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    drifted_profile = {
        "topic_goal": "Investigate automotive sector trends or automated research tools.",
        "scope": [],
        "boundary_notes": [
            "Primary interpretation: Automotive industry and vehicle technology research.",
            "Excluded interpretations: Automated/AI-driven research processes.",
        ],
        "primary_interpretation": "Automotive industry and vehicle technology research.",
        "excluded_interpretations": ["Automated/AI-driven research processes"],
        "disambiguation_hints": [],
        "negative_terms": [],
        "sub_questions": ["How is autonomous driving research changing vehicle AI?"],
        "priorities": [],
        "review_hints": [],
    }
    output_root = tmp_path / "runs"
    run_id = "stage02-primary-repair"
    write_confirmed_topic_intent(
        output_root / run_id,
        topic="auto research",
        excluded_meanings=["automotive research", "self-driving vehicles", "vehicle AI"],
    )

    with _fake_stage02_profile_and_stage03_fallback(monkeypatch, drifted_profile):
        exit_code = cli.main(
            [
                "topic-search",
                "--run-id",
                run_id,
                "--output-root",
                str(output_root),
                "--resume",
                "--plan-only",
            ]
        )

    assert exit_code == 0
    profile = _read_yaml(output_root / run_id / "stage-02" / "topic_profile.md")
    assert "autonomously search and synthesize research literature" in profile["primary_interpretation"]
    assert "automotive" not in profile["primary_interpretation"].casefold()
    assert "AI agent systems that autonomously search and synthesize research literature" not in profile["excluded_interpretations"]
    assert "automotive" not in " ".join(profile["sub_questions"]).casefold()
    assert "vehicle ai" not in " ".join(profile["sub_questions"]).casefold()
    assert "primary_interpretation_repaired_from_confirmed_intent" in profile["quality_warnings"]
    assert "excluded_interpretation_repaired_from_confirmed_intent" in profile["quality_warnings"]


@contextmanager
def _fake_llm_responses(monkeypatch: pytest.MonkeyPatch, payloads: list[dict | str]):  # noqa: ANN001
    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, payload: dict | str) -> None:
            self._payload = payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            content = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
            return content.encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        index = min(calls["count"], len(payloads) - 1)
        calls["count"] += 1
        return FakeResponse(payloads[index])

    from paper_collect import topic_search

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fake_urlopen)
    yield


def test_stage03_deterministic_fallback_keeps_topic_guardrails_and_profile_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    stage02_profile = {
        "topic_goal": "auto research",
        "scope": ["software agents"],
        "negative_terms": ["factory automation"],
        "disambiguation_hints": ["software agents"],
        "review_hints": ["check for autonomous research wording"],
    }
    stage03_invalid_plan = {
        "search_plan_yaml": "{\"topic\": \"auto research\"}",
    }
    output_root = tmp_path / "runs"
    run_id = "stage03-fallback-merge"
    write_confirmed_topic_intent(
        output_root / run_id,
        topic="auto research",
        excluded_meanings=["factory automation"],
    )

    with _fake_llm_responses(
        monkeypatch,
        [
            {"choices": [{"message": {"content": json.dumps({"topic_profile": stage02_profile})}}]},
            {"choices": [{"message": {"content": json.dumps(stage03_invalid_plan)}}]},
        ],
    ):
        exit_code = cli.main(
            [
                "topic-search",
                "--run-id",
                run_id,
                "--output-root",
                str(output_root),
                "--resume",
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    run_root = output_root / run_id
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    run_meta = _read_json(run_root / "run_meta.json")
    assert run_meta["topic_profile_generation"] == {
        "mode": "llm",
        "fallback_reason": "",
    }
    assert search_plan["query_generation"]["mode"] == "deterministic_fallback"
    assert set(search_plan["search_strategies"][0]["negative_terms"]) >= {"factory automation"}
    assert "automobile" not in json.dumps(search_plan).casefold()
    assert "software agents" in search_plan["search_strategies"][0]["disambiguation"]


def test_stage02_sparse_profile_hints_are_merged_into_stage03_fallback_and_surface_in_run_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    sparse_profile = {
        "topic_goal": "auto research",
        "negative_terms": ["research staffing"],
        "disambiguation_hints": ["software agent"],
    }
    output_root = tmp_path / "runs"
    run_id = "stage02-merge"
    write_confirmed_topic_intent(
        output_root / run_id,
        topic="auto research",
        excluded_meanings=["research staffing"],
    )

    with _fake_stage02_profile_and_stage03_fallback(monkeypatch, sparse_profile):
        exit_code = cli.main(
            [
                "topic-search",
                "--run-id",
                run_id,
                "--output-root",
                str(output_root),
                "--resume",
                "--plan-only",
                "--allow-deterministic-fallback",
            ]
        )

    assert exit_code == 0
    run_root = output_root / run_id
    run_meta = _read_json(run_root / "run_meta.json")
    profile = _read_yaml(run_root / "stage-02" / "topic_profile.md")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")

    assert run_meta["topic_profile_generation"] == {
        "mode": "llm",
        "fallback_reason": "",
    }
    assert profile["generation"]["mode"] == "llm"
    assert search_plan["query_generation"]["mode"] == "deterministic_fallback"
    assert "research staffing" in search_plan["negative_terms"]
    assert "software agent" in search_plan["disambiguation"]
    assert "autonomous research agents" in search_plan["disambiguation"]
