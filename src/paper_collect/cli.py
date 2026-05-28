from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from paper_collect.topic_search import run_topic_search

DEFAULT_OUTPUT_ROOT = Path("runtime_data") / "topic_search_runs"


def main(argv: list[str] | None = None) -> int:
    _load_workspace_env()
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "topic-search":
        try:
            result = run_topic_search(
                topic=args.topic,
                description=args.description,
                run_id=args.run_id,
                output_root=args.output_root,
                max_results_per_query=args.max_results_per_query,
                year_min=args.year_min,
                s2_api_key=args.s2_api_key or os.environ.get("S2_API_KEY", ""),
                inter_query_delay=args.inter_query_delay,
                inter_verify_delay=args.inter_verify_delay,
                verification_enabled=not args.disable_verification,
                plan_only=args.plan_only,
                allow_deterministic_fallback=args.allow_deterministic_fallback,
                resume=args.resume,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"run_id: {result.run_id}")
        print(f"run_meta: {result.run_meta_path}")
        if (result.stage03_dir / "search_plan.yaml").exists():
            print(f"stage-03/search_plan.yaml: {result.stage03_dir / 'search_plan.yaml'}")
        elif (result.run_root / "stage-01" / "topic_intent.md").exists():
            print(f"stage-01/topic_intent.md: {result.run_root / 'stage-01' / 'topic_intent.md'}")
        elif (result.run_root / "stage-01" / "topic_intent_draft.md").exists():
            print(f"stage-01/topic_intent_draft.md: {result.run_root / 'stage-01' / 'topic_intent_draft.md'}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paper-collect",
        description="Run the paper-collect topic-only live search runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    topic_parser = subparsers.add_parser(
        "topic-search",
        help="Run topic exploration through the staged topic-search runtime.",
    )
    topic_parser.add_argument("--topic")
    topic_parser.add_argument("--description", default="")
    topic_parser.add_argument("--run-id")
    topic_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    topic_parser.add_argument("--max-results-per-query", type=int, default=40)
    topic_parser.add_argument("--year-min", type=int, default=2020)
    topic_parser.add_argument("--s2-api-key", default="")
    topic_parser.add_argument("--inter-query-delay", type=float, default=1.5)
    topic_parser.add_argument("--inter-verify-delay", type=float, default=1.0)
    topic_parser.add_argument(
        "--disable-verification",
        "--search-only",
        "--skip-verification",
        dest="disable_verification",
        action="store_true",
        help="Stop after Stage 04 and skip Stage 05 verification.",
    )
    topic_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Run Stage 03 only and skip provider execution and verification.",
    )
    topic_parser.add_argument(
        "--allow-deterministic-fallback",
        action="store_true",
        help="Compatibility flag; deterministic fallback is used automatically when LLM planning is unavailable or invalid.",
    )
    topic_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing topic-search run after human-confirmed Stage 01 intent.",
    )
    return parser.parse_args(argv)


def _load_workspace_env() -> None:
    workspace_env = Path(__file__).resolve().parents[2] / ".env"
    cwd_env = Path.cwd() / ".env"
    for path in (workspace_env, cwd_env):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")


if __name__ == "__main__":
    raise SystemExit(main())
