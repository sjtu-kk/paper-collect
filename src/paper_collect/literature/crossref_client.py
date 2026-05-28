# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""Crossref API client."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

_BASE_URL = "https://api.crossref.org/works"
_MAX_ROWS = 50
_TIMEOUT_SEC = 20


def search_crossref(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    rows = max(1, min(limit, _MAX_ROWS))
    params: dict[str, str] = {
        "query": query,
        "rows": str(rows),
    }
    if year_min > 0:
        params["filter"] = f"from-pub-date:{year_min}-01-01"

    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    data = _request_json(url)
    message = data.get("message", {})
    if not isinstance(message, dict):
        return []
    items = message.get("items", [])
    if not isinstance(items, list):
        return []
    papers: list[Paper] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            papers.append(_parse_crossref_work(item))
        except Exception:  # noqa: BLE001
            continue
    return papers


def _request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "paper-collect/0.1; crossref search",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderBlocked("crossref", "rate_limited", "Crossref rate limited") from exc
        raise ProviderBlocked("crossref", "provider_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ProviderBlocked("crossref", "network_failed", str(exc)) from exc
    except TimeoutError as exc:
        raise ProviderBlocked("crossref", "timeout", "Crossref request timed out") from exc
    except json.JSONDecodeError as exc:
        raise ProviderBlocked("crossref", "provider_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProviderBlocked("crossref", "provider_error", "non-object JSON payload")
    return payload


def _parse_crossref_work(item: dict[str, Any]) -> Paper:
    doi = _clean_doi(str(item.get("DOI") or ""))
    title = _first_text(item.get("title")) or doi
    authors = tuple(_parse_authors(item.get("author")))
    year = _issued_year(item)
    container_title = _first_text(item.get("container-title"))
    url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else ""))
    source_metadata = _crossref_authority_metadata(
        doi=doi,
        container_title=container_title,
        issn=item.get("ISSN"),
        publisher=str(item.get("publisher") or ""),
        work_type=str(item.get("type") or ""),
    )
    return Paper(
        paper_id=doi or url or title,
        title=title,
        authors=authors,
        year=year,
        abstract=_strip_markup(str(item.get("abstract") or "")),
        venue=container_title,
        citation_count=int(item.get("is-referenced-by-count") or 0),
        doi=doi,
        arxiv_id="",
        url=url,
        source="crossref",
        source_metadata=source_metadata,
    )


def _parse_authors(raw_authors: Any) -> list[Author]:
    if not isinstance(raw_authors, list):
        return []
    authors: list[Author] = []
    for raw_author in raw_authors:
        if not isinstance(raw_author, dict):
            continue
        given = str(raw_author.get("given") or "").strip()
        family = str(raw_author.get("family") or "").strip()
        name = " ".join(part for part in (given, family) if part).strip()
        if name:
            authors.append(Author(name=name))
    return authors


def _issued_year(item: dict[str, Any]) -> int:
    for key in ("issued", "published-print", "published-online", "created"):
        raw = item.get(key)
        if not isinstance(raw, dict):
            continue
        date_parts = raw.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts:
            continue
        first = date_parts[0]
        if not isinstance(first, list) or not first:
            continue
        try:
            return int(first[0])
        except (TypeError, ValueError):
            continue
    return 0


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return _strip_markup(text)
        return ""
    return _strip_markup(str(value or "").strip())


def _crossref_authority_metadata(
    *,
    doi: str,
    container_title: str,
    issn: Any,
    publisher: str,
    work_type: str,
) -> dict[str, Any]:
    issn_values = [str(item).strip() for item in issn if str(item).strip()] if isinstance(issn, list) else []
    metadata = {
        "evidence_kind": "crossref_bibliographic_authority",
        "provider_id": "crossref",
        "crossref_id": doi,
        "container_title": container_title,
        "issn": issn_values,
        "publisher": publisher,
        "type": work_type,
    }
    return metadata


def _clean_doi(value: str) -> str:
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            return doi[len(prefix) :]
    return doi


def _strip_markup(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return " ".join(text.split())
