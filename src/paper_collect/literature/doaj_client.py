# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""DOAJ API client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

_BASE_URL = "https://doaj.org/api/v4/search/articles"
_MAX_PAGE_SIZE = 50
_TIMEOUT_SEC = 20


def search_doaj(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    page_size = max(1, min(limit, _MAX_PAGE_SIZE))
    url = f"{_BASE_URL}/{urllib.parse.quote(query, safe='')}?{urllib.parse.urlencode({'page': '1', 'pageSize': str(page_size)})}"
    data = _request_json(url)
    results = data.get("results", [])
    if not isinstance(results, list):
        return []
    papers: list[Paper] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            paper = _parse_doaj_article(item)
        except Exception:  # noqa: BLE001
            continue
        if year_min and paper.year and paper.year < year_min:
            continue
        papers.append(paper)
    return papers


def _request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "paper-collect/0.1; doaj search",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderBlocked("doaj", "rate_limited", "DOAJ rate limited") from exc
        raise ProviderBlocked("doaj", "provider_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ProviderBlocked("doaj", "network_failed", str(exc)) from exc
    except TimeoutError as exc:
        raise ProviderBlocked("doaj", "timeout", "DOAJ request timed out") from exc
    except json.JSONDecodeError as exc:
        raise ProviderBlocked("doaj", "provider_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProviderBlocked("doaj", "provider_error", "non-object JSON payload")
    return payload


def _parse_doaj_article(item: dict[str, Any]) -> Paper:
    bibjson = item.get("bibjson", {})
    if not isinstance(bibjson, dict):
        bibjson = {}
    doaj_id = str(item.get("id") or "")
    identifiers = bibjson.get("identifier", [])
    doi = _identifier_by_type(identifiers, "doi")
    journal = bibjson.get("journal", {})
    if not isinstance(journal, dict):
        journal = {}
    journal_title = str(journal.get("title") or "")
    issn = _issn_values(identifiers, journal.get("issns"))
    publisher = str(journal.get("publisher") or "")
    language = _string_list(journal.get("language"))
    license_values = _license_values(bibjson.get("license"))
    link = _first_link(bibjson.get("link"))
    link_url = str(link.get("url") or "")
    link_type = str(link.get("type") or "")
    title = str(bibjson.get("title") or doaj_id or doi)
    return Paper(
        paper_id=doaj_id or doi or link_url or title,
        title=title,
        authors=tuple(_parse_authors(bibjson.get("author"))),
        year=_as_int(bibjson.get("year")),
        abstract=str(bibjson.get("abstract") or ""),
        venue=journal_title,
        citation_count=0,
        doi=doi,
        arxiv_id="",
        url=link_url or (f"https://doi.org/{doi}" if doi else ""),
        source="doaj",
        source_metadata={
            "evidence_kind": "doaj_oa_metadata",
            "provider_id": "doaj",
            "doaj_id": doaj_id,
            "journal_title": journal_title,
            "issn": issn,
            "publisher": publisher,
            "license": license_values,
            "language": language,
            "link_url": link_url,
            "link_type": link_type,
        },
    )


def _parse_authors(raw_authors: Any) -> list[Author]:
    if not isinstance(raw_authors, list):
        return []
    authors: list[Author] = []
    for raw_author in raw_authors:
        if not isinstance(raw_author, dict):
            continue
        name = str(raw_author.get("name") or "").strip()
        affiliation = str(raw_author.get("affiliation") or "").strip()
        if name:
            authors.append(Author(name=name, affiliation=affiliation))
    return authors


def _identifier_by_type(raw_identifiers: Any, identifier_type: str) -> str:
    if not isinstance(raw_identifiers, list):
        return ""
    target = identifier_type.casefold()
    for item in raw_identifiers:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").casefold() == target:
            return str(item.get("id") or "").strip()
    return ""


def _issn_values(raw_identifiers: Any, journal_issns: Any) -> list[str]:
    values: list[str] = []
    for item in _string_list(journal_issns):
        if item not in values:
            values.append(item)
    if isinstance(raw_identifiers, list):
        for item in raw_identifiers:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").casefold() in {"issn", "eissn", "pissn"}:
                value = str(item.get("id") or "").strip()
                if value and value not in values:
                    values.append(value)
    return values


def _license_values(raw_license: Any) -> list[str]:
    if isinstance(raw_license, list):
        values: list[str] = []
        for item in raw_license:
            if isinstance(item, dict):
                value = str(item.get("title") or item.get("type") or "").strip()
            else:
                value = str(item or "").strip()
            if value and value not in values:
                values.append(value)
        return values
    if isinstance(raw_license, dict):
        value = str(raw_license.get("title") or raw_license.get("type") or "").strip()
        return [value] if value else []
    value = str(raw_license or "").strip()
    return [value] if value else []


def _first_link(raw_links: Any) -> dict[str, Any]:
    if isinstance(raw_links, list):
        for item in raw_links:
            if isinstance(item, dict) and item.get("url"):
                return item
    return {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
