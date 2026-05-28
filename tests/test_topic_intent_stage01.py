from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from paper_collect import cli


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


def _write_topic_intent(
    run_root: Path,
    *,
    topic: str,
    description: str = "",
    status: str = "confirmed",
    human_notes: str = "confirmed by test",
) -> Path:
    intent = {
        "artifact_version": "topic_search_topic_intent.v1",
        "status": status,
        "topic": topic,
        "description": description,
        "intended_meaning": ["AI agent systems that autonomously search and synthesize research literature"],
        "research_purpose": ["Learn how others automate scholarly information collection"],
        "included_concepts": ["autonomous research agents", "automated literature discovery"],
        "excluded_meanings": ["automotive research", "self-driving vehicles"],
        "uncertainties": [],
        "human_notes": human_notes,
    }
    stage01_dir = run_root / "stage-01"
    stage01_dir.mkdir(parents=True, exist_ok=True)
    intent_path = stage01_dir / "topic_intent.md"
    intent_path.write_text(
        yaml.safe_dump(intent, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return intent_path


def _write_confirmed_intent(run_root: Path, *, topic: str, description: str = "") -> None:
    _write_topic_intent(run_root, topic=topic, description=description, status="confirmed")


@contextmanager
def _fake_llm_response(monkeypatch: pytest.MonkeyPatch, payload: dict | str):  # noqa: ANN001
    captured: dict = {}

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
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return FakeResponse()

    from paper_collect import topic_search

    monkeypatch.setattr(topic_search.urllib.request, "urlopen", fake_urlopen)
    yield captured


def test_new_topic_search_run_blocks_after_stage01_draft_until_human_confirms(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "auto research",
            "--description",
            "AI agent autonomous research systems for learning automated scholarly collection",
            "--run-id",
            "intent-gate",
            "--output-root",
            str(output_root),
            "--plan-only",
        ]
    )

    assert exit_code == 0
    run_root = output_root / "intent-gate"
    draft = _read_yaml(run_root / "stage-01" / "topic_intent_draft.md")
    formal = _read_yaml(run_root / "stage-01" / "topic_intent.md")
    run_meta = _read_json(run_root / "run_meta.json")

    assert draft["artifact_version"] == "topic_search_topic_intent.v1"
    assert draft["status"] == "draft"
    assert formal["artifact_version"] == "topic_search_topic_intent.v1"
    assert formal["status"] == "draft"
    assert formal == draft
    assert draft["topic"] == "auto research"
    assert draft["description"] == "AI agent autonomous research systems for learning automated scholarly collection"
    assert draft["intended_meaning"]
    assert "research_purpose" in draft
    assert "included_concepts" in draft
    assert "excluded_meanings" in draft
    assert "uncertainties" in draft
    assert "human" in str(draft["human_notes"]).casefold()
    assert "copy this file" not in str(draft["human_notes"]).casefold()
    assert "change status to confirmed" in str(draft["human_notes"]).casefold()
    assert run_meta["status"] == "needs_human_topic_intent"
    assert run_meta["stage_statuses"] == {
        "stage-01": "needs_human_topic_intent",
        "stage-02": "blocked",
        "stage-03": "blocked",
        "stage-04": "blocked",
        "stage-05": "blocked",
    }
    assert not (run_root / "stage-02").exists()
    assert not (run_root / "stage-03").exists()
    assert not (run_root / "stage-04").exists()


def test_plan_only_rerun_with_existing_formal_draft_reblocks_without_overwriting_formal(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"
    run_root = output_root / "intent-formal-draft"
    intent_path = _write_topic_intent(
        run_root,
        topic="auto research",
        status="draft",
        human_notes="sentinel human draft notes",
    )
    original_formal_text = intent_path.read_text(encoding="utf-8")

    exit_code = cli.main(
        [
            "topic-search",
            "--topic",
            "auto research",
            "--run-id",
            "intent-formal-draft",
            "--output-root",
            str(output_root),
            "--plan-only",
        ]
    )

    run_meta = _read_json(run_root / "run_meta.json")
    assert exit_code == 0
    assert run_meta["status"] == "needs_human_topic_intent"
    assert run_meta["stage_statuses"]["stage-02"] == "blocked"
    assert intent_path.read_text(encoding="utf-8") == original_formal_text
    assert _read_yaml(run_root / "stage-01" / "topic_intent_draft.md")["status"] == "draft"
    assert not (run_root / "stage-02").exists()


def test_resume_with_existing_formal_draft_reblocks_and_reconstructs_missing_draft(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"
    run_root = output_root / "intent-resume-draft"
    intent_path = _write_topic_intent(
        run_root,
        topic="auto research",
        status="draft",
        human_notes="sentinel resume draft notes",
    )
    original_formal_text = intent_path.read_text(encoding="utf-8")

    exit_code = cli.main(
        [
            "topic-search",
            "--run-id",
            "intent-resume-draft",
            "--output-root",
            str(output_root),
            "--resume",
            "--plan-only",
        ]
    )

    run_meta = _read_json(run_root / "run_meta.json")
    draft = _read_yaml(run_root / "stage-01" / "topic_intent_draft.md")
    assert exit_code == 0
    assert run_meta["status"] == "needs_human_topic_intent"
    assert run_meta["stage_statuses"]["stage-02"] == "blocked"
    assert intent_path.read_text(encoding="utf-8") == original_formal_text
    assert draft["status"] == "draft"
    assert draft["human_notes"] == "Draft reconstructed from topic_intent.md for review history."
    assert not (run_root / "stage-02").exists()


def test_stage01_draft_uses_llm_when_configured_and_keeps_editable_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    output_root = tmp_path / "runs"
    llm_payload = {
        "topic_intent_draft": {
            "intended_meaning": [
                "AI systems that autonomously search, read, and synthesize research literature"
            ],
            "research_purpose": ["Learn how agent systems automate scholarly collection"],
            "included_concepts": ["autonomous research agents", "literature discovery agents"],
            "excluded_meanings": ["automotive research", "market research automation"],
            "uncertainties": ["Whether to include scientific hypothesis generation systems"],
            "human_notes": "Delete wrong candidates, then change status to confirmed.",
            "sub_questions": ["should not leak stage 02 content"],
            "queries": ["should not leak stage 03 content"],
        }
    }

    with _fake_llm_response(monkeypatch, llm_payload) as captured:
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "auto research",
                "--description",
                "AI agent autonomous research systems for learning automated scholarly collection",
                "--run-id",
                "intent-llm-draft",
                "--output-root",
                str(output_root),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    draft = _read_yaml(output_root / "intent-llm-draft" / "stage-01" / "topic_intent_draft.md")
    prompt = captured["body"]["messages"][1]["content"]

    assert captured["url"] == "https://llm.example.test/v1/chat/completions"
    assert "Stage 01-lite topic intent draft" in prompt
    assert "auto research" in prompt
    assert "AI agent autonomous research systems" in prompt
    assert "copy to topic_intent.md" not in prompt
    assert draft["artifact_version"] == "topic_search_topic_intent.v1"
    assert draft["status"] == "draft"
    assert draft["topic"] == "auto research"
    assert draft["description"] == "AI agent autonomous research systems for learning automated scholarly collection"
    assert draft["intended_meaning"] == [
        "AI systems that autonomously search, read, and synthesize research literature"
    ]
    assert draft["research_purpose"] == ["Learn how agent systems automate scholarly collection"]
    assert draft["included_concepts"] == ["autonomous research agents", "literature discovery agents"]
    assert draft["excluded_meanings"] == ["automotive research", "market research automation"]
    assert draft["uncertainties"] == ["Whether to include scientific hypothesis generation systems"]
    assert "human" in draft["human_notes"].casefold()
    assert "sub_questions" not in draft
    assert "queries" not in draft
    assert not (output_root / "intent-llm-draft" / "stage-02").exists()


def test_stage01_llm_draft_accepts_string_fields_and_keeps_unicode_editable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    output_root = tmp_path / "runs"
    llm_payload = {
        "topic_intent_draft": {
            "intended_meaning": "AI agents that autonomously collect and synthesize scholarly literature",
            "research_purpose": "学习自动搜集期刊和会议等科研信息的 agent 系统",
            "included_concepts": [
                "autonomous research agents",
                "automated scholarly information collection",
            ],
            "excluded_meanings": "automotive research",
            "uncertainties": "Whether to include self-driving labs or only literature-search agents",
            "human_notes": "Human should delete irrelevant meanings and confirm.",
        }
    }

    with _fake_llm_response(monkeypatch, llm_payload):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "auto research",
                "--description",
                "是 AI agent autonomous research 的相关工作",
                "--run-id",
                "intent-llm-string-draft",
                "--output-root",
                str(output_root),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    draft_path = output_root / "intent-llm-string-draft" / "stage-01" / "topic_intent_draft.md"
    draft_text = draft_path.read_text(encoding="utf-8")
    draft = _read_yaml(draft_path)

    assert draft["intended_meaning"] == [
        "AI agents that autonomously collect and synthesize scholarly literature"
    ]
    assert draft["research_purpose"] == ["学习自动搜集期刊和会议等科研信息的 agent 系统"]
    assert draft["excluded_meanings"] == ["automotive research"]
    assert draft["uncertainties"] == [
        "Whether to include self-driving labs or only literature-search agents"
    ]
    assert "学习自动搜集期刊" in draft_text
    assert "\\u5b66" not in draft_text


def test_stage01_llm_draft_keeps_complete_editable_intent_sentences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    output_root = tmp_path / "runs"
    long_meaning = (
        "Literature and systems related to AI agents performing autonomous scientific research, "
        "with a specific focus on the automated gathering, filtering, and processing of academic "
        "publications including journals, conferences, and preprints."
    )
    llm_payload = {
        "topic_intent_draft": {
            "intended_meaning": long_meaning,
            "research_purpose": long_meaning,
            "included_concepts": ["automated literature retrieval"],
            "excluded_meanings": ["automotive research"],
            "uncertainties": [],
            "human_notes": "Human should confirm.",
        }
    }

    with _fake_llm_response(monkeypatch, llm_payload):
        exit_code = cli.main(
            [
                "topic-search",
                "--topic",
                "auto research",
                "--run-id",
                "intent-llm-complete-sentence",
                "--output-root",
                str(output_root),
                "--plan-only",
            ]
        )

    assert exit_code == 0
    draft = _read_yaml(output_root / "intent-llm-complete-sentence" / "stage-01" / "topic_intent_draft.md")

    assert draft["intended_meaning"] == [long_meaning]
    assert draft["research_purpose"] == [long_meaning]


def test_resume_with_confirmed_intent_continues_from_stage02_without_topic(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"
    run_root = output_root / "intent-resume"
    _write_confirmed_intent(run_root, topic="auto research")

    exit_code = cli.main(
        [
            "topic-search",
            "--run-id",
            "intent-resume",
            "--output-root",
            str(output_root),
            "--resume",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    run_meta = _read_json(run_root / "run_meta.json")
    profile = _read_yaml(run_root / "stage-02" / "topic_profile.md")
    search_plan = _read_yaml(run_root / "stage-03" / "search_plan.yaml")
    assert run_meta["request"]["topic"] == "auto research"
    assert run_meta["stage_statuses"]["stage-01"] == "confirmed"
    assert run_meta["stage_statuses"]["stage-02"] == "complete"
    assert run_meta["stage_statuses"]["stage-03"] == "complete"
    assert profile["topic"] == "auto research"
    assert search_plan["topic_profile_ref"] == "../stage-02/topic_profile.md"


def test_resume_rejects_conflicting_topic_or_description(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "runs"
    run_root = output_root / "intent-conflict"
    _write_confirmed_intent(run_root, topic="auto research", description="agent research")

    exit_code = cli.main(
        [
            "topic-search",
            "--run-id",
            "intent-conflict",
            "--output-root",
            str(output_root),
            "--resume",
            "--topic",
            "automotive research",
        ]
    )

    assert exit_code == 2
    assert "does not match existing run topic" in capsys.readouterr().err
    assert not (run_root / "stage-02").exists()


@pytest.mark.parametrize(
    ("intent_status", "intended_meaning", "expected_error"),
    [
        ("confirmed", [], "non-empty intended_meaning"),
    ],
)
def test_resume_blocks_unconfirmed_or_empty_intent_before_stage02(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    intent_status: str,
    intended_meaning: list[str],
    expected_error: str,
) -> None:
    output_root = tmp_path / "runs"
    run_root = output_root / "intent-invalid"
    _write_confirmed_intent(run_root, topic="auto research")
    intent_path = run_root / "stage-01" / "topic_intent.md"
    intent = _read_yaml(intent_path)
    intent["status"] = intent_status
    intent["intended_meaning"] = intended_meaning
    intent_path.write_text(
        yaml.safe_dump(intent, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "topic-search",
            "--run-id",
            "intent-invalid",
            "--output-root",
            str(output_root),
            "--resume",
            "--plan-only",
        ]
    )

    assert exit_code == 2
    assert expected_error in capsys.readouterr().err
    assert not (run_root / "stage-02").exists()


def test_resume_requires_confirmed_intent_file_before_stage02(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "runs"
    run_root = output_root / "intent-missing"
    (run_root / "stage-01").mkdir(parents=True)

    exit_code = cli.main(
        [
            "topic-search",
            "--run-id",
            "intent-missing",
            "--output-root",
            str(output_root),
            "--resume",
            "--plan-only",
        ]
    )

    assert exit_code == 2
    assert "confirmed topic intent is required" in capsys.readouterr().err
    assert not (run_root / "stage-02").exists()
