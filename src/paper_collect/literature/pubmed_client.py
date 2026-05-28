# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""PubMed E-utilities client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MAX_RETMAX = 50
_TIMEOUT_SEC = 20


def search_pubmed(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
) -> list[Paper]:
    retmax = max(1, min(limit, _MAX_RETMAX))
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
    }
    if year_min > 0:
        search_params["datetype"] = "pdat"
        search_params["mindate"] = str(year_min)
    search_url = f"{_EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(search_params)}"
    search_payload = _request_json(search_url)
    id_list = search_payload.get("esearchresult", {}).get("idlist", [])
    if not isinstance(id_list, list):
        return []
    pmids = [str(item).strip() for item in id_list if str(item).strip()]
    if not pmids:
        return []
    summary_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    summary_url = f"{_EUTILS_BASE}/esummary.fcgi?{urllib.parse.urlencode(summary_params)}"
    summary_payload = _request_json(summary_url)
    result = summary_payload.get("result", {})
    if not isinstance(result, dict):
        return []
    papers: list[Paper] = []
    for pmid in pmids:
        item = result.get(pmid)
        if not isinstance(item, dict):
            continue
        try:
            paper = _parse_pubmed_summary(item)
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
            "User-Agent": "paper-collect/0.1; pubmed search",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ProviderBlocked("pubmed", "rate_limited", "PubMed rate limited") from exc
        raise ProviderBlocked("pubmed", "provider_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ProviderBlocked("pubmed", "network_failed", str(exc)) from exc
    except TimeoutError as exc:
        raise ProviderBlocked("pubmed", "timeout", "PubMed request timed out") from exc
    except json.JSONDecodeError as exc:
        raise ProviderBlocked("pubmed", "provider_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProviderBlocked("pubmed", "provider_error", "non-object JSON payload")
    return payload


def _parse_pubmed_summary(item: dict[str, Any]) -> Paper:
    pmid = str(item.get("uid") or "")
    doi = _article_id(item.get("articleids"), "doi")
    journal = str(item.get("fulljournalname") or item.get("source") or "")
    source = str(item.get("source") or "")
    publication_types = _string_list(item.get("pubtype"))
    language = _string_list(item.get("lang"))
    attributes = _string_list(item.get("attributes"))
    has_abstract = any(value.casefold() == "has abstract" for value in attributes)
    return Paper(
        paper_id=pmid or doi or str(item.get("title") or ""),
        title=str(item.get("title") or pmid),
        authors=tuple(_parse_authors(item.get("authors"))),
        year=_pub_year(str(item.get("pubdate") or item.get("sortpubdate") or "")),
        abstract="Has Abstract" if has_abstract else "",
        venue=journal,
        citation_count=0,
        doi=doi,
        arxiv_id="",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        source="pubmed",
        source_metadata={
            "evidence_kind": "pubmed_biomedical_corpus",
            "provider_id": "pubmed",
            "pmid": pmid,
            "journal": journal,
            "source": source,
            "publication_types": publication_types,
            "language": language,
            "has_abstract": has_abstract,
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
        if name:
            authors.append(Author(name=name))
    return authors


def _article_id(raw_ids: Any, id_type: str) -> str:
    if not isinstance(raw_ids, list):
        return ""
    target = id_type.casefold()
    for item in raw_ids:
        if not isinstance(item, dict):
            continue
        if str(item.get("idtype") or "").casefold() == target:
            return str(item.get("value") or "").strip()
    return ""


def _pub_year(value: str) -> int:
    for token in value.replace("/", " ").split():
        if len(token) >= 4 and token[:4].isdigit():
            return int(token[:4])
    return 0


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
