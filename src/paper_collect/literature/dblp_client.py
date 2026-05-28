# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""DBLP API client."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

_BASE_URL = "https://dblp.org/search/publ/api"
_MAX_HITS = 50
_TIMEOUT_SEC = 20
_COMPACT_MAX_TERMS = 7
_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "with",
        "without",
        "survey",
        "framing",
        "provider",
        "prose",
        "irrelevant",
    }
)


def search_dblp(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    hits = max(1, min(limit, _MAX_HITS))
    provider_query = compact_dblp_query(query)
    params = {
        "q": provider_query,
        "h": str(hits),
        "format": "json",
    }
    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    data = _request_json(url)
    raw_hits = data.get("result", {}).get("hits", {}).get("hit", [])
    if isinstance(raw_hits, dict):
        raw_hits = [raw_hits]
    if not isinstance(raw_hits, list):
        return []
    papers: list[Paper] = []
    for raw_hit in raw_hits:
        if not isinstance(raw_hit, dict):
            continue
        try:
            paper = _parse_dblp_hit(raw_hit)
        except Exception:  # noqa: BLE001
            continue
        if year_min and paper.year and paper.year < year_min:
            continue
        papers.append(paper)
    return papers


def compact_dblp_query(query: str) -> str:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)
    compact_terms = [term for term in terms if term.casefold() not in _STOP_TERMS]
    compact = " ".join(compact_terms[:_COMPACT_MAX_TERMS]).strip()
    return compact or " ".join(terms[:_COMPACT_MAX_TERMS]).strip() or query.strip()


def dblp_query_summary(query: str) -> dict[str, str]:
    provider_query = compact_dblp_query(query)
    summary = {
        "original_query_text": query,
        "provider_query_text": provider_query,
    }
    if provider_query != query:
        summary["rewrite_reason"] = "dblp_keyword_compaction"
    return summary


def _request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "paper-collect/0.1; dblp search",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderBlocked("dblp", "rate_limited", "DBLP rate limited") from exc
        raise ProviderBlocked("dblp", "provider_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ProviderBlocked("dblp", "network_failed", str(exc)) from exc
    except TimeoutError as exc:
        raise ProviderBlocked("dblp", "timeout", "DBLP request timed out") from exc
    except json.JSONDecodeError as exc:
        raise ProviderBlocked("dblp", "provider_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProviderBlocked("dblp", "provider_error", "non-object JSON payload")
    return payload


def _parse_dblp_hit(raw_hit: dict[str, Any]) -> Paper:
    info = raw_hit.get("info", {})
    if not isinstance(info, dict):
        info = {}
    dblp_key = str(info.get("key") or "")
    title = str(info.get("title") or dblp_key)
    venue = str(info.get("venue") or "")
    publication_type = str(info.get("type") or "")
    ee = _first_value(info.get("ee"))
    url = str(info.get("url") or "")
    doi = _clean_doi(str(info.get("doi") or ""))
    return Paper(
        paper_id=dblp_key or url or title,
        title=title,
        authors=tuple(_parse_authors(info.get("authors"))),
        year=_as_int(info.get("year")),
        abstract="",
        venue=venue,
        citation_count=0,
        doi=doi,
        arxiv_id="",
        url=url or ee,
        source="dblp",
        source_metadata={
            "evidence_kind": "dblp_cs_bibliographic",
            "provider_id": "dblp",
            "dblp_key": dblp_key,
            "venue": venue,
            "publication_type": publication_type,
            "ee": ee,
        },
    )


def _parse_authors(raw_authors: Any) -> list[Author]:
    author_items = raw_authors.get("author", []) if isinstance(raw_authors, dict) else []
    if isinstance(author_items, dict):
        author_items = [author_items]
    if isinstance(author_items, str):
        author_items = [author_items]
    if not isinstance(author_items, list):
        return []
    authors: list[Author] = []
    for raw_author in author_items:
        if isinstance(raw_author, dict):
            name = str(raw_author.get("text") or "").strip()
        else:
            name = str(raw_author or "").strip()
        if name:
            authors.append(Author(name=name))
    return authors


def _first_value(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean_doi(value: str) -> str:
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            return doi[len(prefix) :]
    return doi
