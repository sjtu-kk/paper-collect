# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""OpenAlex API client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

_BASE_URL = "https://api.openalex.org/works"
_POLITE_EMAIL = "researchclaw@users.noreply.github.com"
_MAX_PER_REQUEST = 50
_TIMEOUT_SEC = 20


def search_openalex(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
    email: str = _POLITE_EMAIL,
) -> list[Paper]:
    limit = max(1, min(limit, _MAX_PER_REQUEST))
    filters = []
    if year_min > 0:
        filters.append(f"from_publication_date:{year_min}-01-01")

    params: dict[str, str] = {
        "search": query,
        "per_page": str(limit),
        "mailto": email,
        "select": "id,title,authorships,publication_year,primary_location,cited_by_count,doi,ids,abstract_inverted_index",
    }
    if filters:
        params["filter"] = ",".join(filters)

    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    data = _request_json(url)
    results = data.get("results", [])
    if not isinstance(results, list):
        return []
    papers: list[Paper] = []
    for item in results:
        try:
            papers.append(_parse_openalex_work(item))
        except Exception:  # noqa: BLE001
            continue
    return papers


def _request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "paper-collect/0.1; openalex search",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderBlocked("openalex", "rate_limited", "OpenAlex rate limited") from exc
        raise ProviderBlocked("openalex", "provider_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ProviderBlocked("openalex", "network_failed", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderBlocked("openalex", "provider_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProviderBlocked("openalex", "provider_error", "non-object JSON payload")
    return payload


def _parse_openalex_work(item: dict[str, Any]) -> Paper:
    identifiers = item.get("ids") or {}
    authors = []
    for authorship in item.get("authorships", []) or []:
        author_name = ((authorship.get("author") or {}).get("display_name") or "").strip()
        if author_name:
            authors.append(Author(name=author_name))
    title = str(item.get("title") or item.get("display_name") or "")
    abstract = _reconstruct_abstract(item.get("abstract_inverted_index") or {})
    venue = ""
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = str(source.get("display_name") or item.get("host_venue", {}).get("display_name", "") or "")
    source_metadata = _openalex_source_metadata(source, item.get("host_venue"))
    doi = _clean_doi(item.get("doi", "") or "")
    paper_id = str(item.get("id") or doi or title)
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=tuple(authors),
        year=int(item.get("publication_year") or 0) or 0,
        abstract=abstract,
        venue=venue,
        citation_count=int(item.get("cited_by_count") or 0),
        doi=doi,
        arxiv_id="",
        url=str(item.get("id") or ""),
        source="openalex",
        source_metadata=source_metadata,
    )


def _openalex_source_metadata(source: Any, host_venue: Any) -> dict[str, Any]:
    source_payload = source if isinstance(source, dict) else {}
    host_payload = host_venue if isinstance(host_venue, dict) else {}
    source_id = str(source_payload.get("id") or host_payload.get("id") or "").strip()
    display_name = str(source_payload.get("display_name") or host_payload.get("display_name") or "").strip()
    issn_l = str(source_payload.get("issn_l") or host_payload.get("issn_l") or "").strip()
    raw_issn = source_payload.get("issn") or host_payload.get("issn") or []
    issn = [str(value).strip() for value in raw_issn if str(value).strip()] if isinstance(raw_issn, list) else []
    source_type = str(source_payload.get("type") or host_payload.get("type") or "").strip()
    host_organization = str(source_payload.get("host_organization") or host_payload.get("host_organization") or "").strip()
    if not any((source_id, display_name, issn_l, issn, source_type, host_organization)):
        return {}
    return {
        "evidence_kind": "openalex_source_authority",
        "source_id": source_id,
        "display_name": display_name,
        "issn_l": issn_l,
        "issn": issn,
        "type": source_type,
        "host_organization": host_organization,
    }


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, token_positions in inverted_index.items():
        for position in token_positions or []:
            positions.append((int(position), token))
    return " ".join(token for _, token in sorted(positions))


def _clean_doi(value: str) -> str:
    doi = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            return doi[len(prefix) :]
    return doi
