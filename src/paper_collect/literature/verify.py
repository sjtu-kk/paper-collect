# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""Citation verification engine."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from paper_collect.literature.errors import ProviderBlocked
from paper_collect.literature.models import Author, Paper


class VerifyStatus(str, Enum):
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    HALLUCINATED = "hallucinated"
    SKIPPED = "skipped"


@dataclass
class CitationResult:
    cite_key: str
    title: str
    status: VerifyStatus
    confidence: float
    method: str
    details: str = ""
    matched_paper: Paper | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "cite_key": self.cite_key,
            "title": self.title,
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "details": self.details,
        }
        if self.matched_paper is not None:
            data["matched_paper"] = self.matched_paper.to_dict()
        return data


@dataclass
class VerificationReport:
    total: int = 0
    verified: int = 0
    suspicious: int = 0
    hallucinated: int = 0
    skipped: int = 0
    results: list[CitationResult] = field(default_factory=list)

    @property
    def integrity_score(self) -> float:
        verifiable = self.total - self.skipped
        if verifiable <= 0:
            return 1.0
        return round(self.verified / verifiable, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": self.total,
                "verified": self.verified,
                "suspicious": self.suspicious,
                "hallucinated": self.hallucinated,
                "skipped": self.skipped,
                "integrity_score": self.integrity_score,
            },
            "results": [result.to_dict() for result in self.results],
        }


_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\s*\}(?=\s*(?:@|\Z))", re.DOTALL)
_FIELD_RE = re.compile(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.DOTALL)


def parse_bibtex_entries(bib_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for match in _ENTRY_RE.finditer(bib_text):
        entry: dict[str, str] = {"type": match.group(1).lower(), "key": match.group(2).strip()}
        body = match.group(3)
        for field_match in _FIELD_RE.finditer(body):
            entry[field_match.group(1).lower()] = field_match.group(2).strip()
        entries.append(entry)
    return entries


def title_similarity(a: str, b: str) -> float:
    def words(text: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9\s]", "", text.lower()).split()) - {""}

    left = words(a)
    right = words(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def verify_by_arxiv_id(arxiv_id: str, expected_title: str) -> CitationResult | None:
    params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": "1"})
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper-collect/0.1"})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read().decode("utf-8")
    except Exception:
        return None
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    entries = root.findall("atom:entry", {"atom": "http://www.w3.org/2005/Atom"})
    if not entries:
        return CitationResult("", expected_title, VerifyStatus.HALLUCINATED, 0.9, "arxiv_id", f"arXiv ID {arxiv_id} not found in arXiv")
    entry = entries[0]
    found_title = _xml_text(entry, "atom:title")
    entry_id = _xml_text(entry, "atom:id")
    if "api/errors" in entry_id or not found_title or found_title.lower() == "error":
        return CitationResult("", expected_title, VerifyStatus.HALLUCINATED, 0.9, "arxiv_id", f"arXiv ID {arxiv_id} returned error or empty response")
    sim = title_similarity(expected_title, found_title)
    status = VerifyStatus.VERIFIED if sim >= 0.8 else VerifyStatus.SUSPICIOUS
    return CitationResult("", expected_title, status, sim, "arxiv_id", f"Confirmed via arXiv: '{found_title}'" if status == VerifyStatus.VERIFIED else f"arXiv ID exists but title differs (sim={sim:.2f}): '{found_title}'")


def verify_by_doi(doi: str, expected_title: str) -> CitationResult | None:
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{encoded_doi}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper-collect/0.1", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and (doi.startswith("10.48550/") or doi.startswith("10.5281/")):
            dc_result = _verify_doi_datacite(doi, expected_title)
            if dc_result is not None:
                return dc_result
        if exc.code == 404:
            return CitationResult("", expected_title, VerifyStatus.HALLUCINATED, 0.9, "doi", f"DOI {doi} not found (HTTP 404)")
        return None
    except Exception:
        return None

    message = body.get("message", {})
    titles = message.get("title", [])
    found_title = titles[0] if titles else ""
    if not found_title:
        return CitationResult("", expected_title, VerifyStatus.VERIFIED, 0.85, "doi", f"DOI {doi} resolves via CrossRef (no title comparison)")
    sim = title_similarity(expected_title, found_title)
    if sim >= 0.8:
        return CitationResult("", expected_title, VerifyStatus.VERIFIED, sim, "doi", f"Confirmed via CrossRef: '{found_title}'")
    if sim >= 0.5:
        return CitationResult("", expected_title, VerifyStatus.SUSPICIOUS, sim, "doi", f"DOI resolves but title differs (sim={sim:.2f}): '{found_title}'")
    return CitationResult("", expected_title, VerifyStatus.SUSPICIOUS, sim, "doi", f"DOI resolves but title mismatch (sim={sim:.2f}): '{found_title}'")


def _verify_doi_datacite(doi: str, expected_title: str) -> CitationResult | None:
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"https://api.datacite.org/dois/{encoded_doi}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper-collect/0.1", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    attrs = body.get("data", {}).get("attributes", {})
    titles = attrs.get("titles", [])
    found_title = titles[0].get("title", "") if titles else ""
    if not found_title:
        return CitationResult("", expected_title, VerifyStatus.VERIFIED, 0.85, "doi", f"DOI {doi} resolves via DataCite (no title comparison)")
    sim = title_similarity(expected_title, found_title)
    if sim >= 0.8:
        return CitationResult("", expected_title, VerifyStatus.VERIFIED, sim, "doi", f"Confirmed via DataCite: '{found_title}'")
    if sim >= 0.5:
        return CitationResult("", expected_title, VerifyStatus.SUSPICIOUS, sim, "doi", f"DataCite DOI resolves but title differs (sim={sim:.2f}): '{found_title}'")
    return CitationResult("", expected_title, VerifyStatus.SUSPICIOUS, sim, "doi", f"DataCite DOI resolves but title mismatch (sim={sim:.2f}): '{found_title}'")


def verify_by_openalex(title: str) -> CitationResult | None:
    params = urllib.parse.urlencode({"filter": "title.search:" + title.replace(",", " ").replace(":", " "), "per_page": "5", "mailto": "researchclaw@users.noreply.github.com"})
    url = f"https://api.openalex.org/works?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper-collect/0.1", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    results = body.get("results", [])
    if not results:
        return CitationResult("", title, VerifyStatus.HALLUCINATED, 0.7, "openalex", "No results found via OpenAlex")
    best_sim = 0.0
    best_title = ""
    for record in results:
        found_title = str(record.get("title", ""))
        if found_title:
            sim = title_similarity(title, found_title)
            if sim > best_sim:
                best_sim = sim
                best_title = found_title
    if best_sim >= 0.8:
        return CitationResult("", title, VerifyStatus.VERIFIED, best_sim, "openalex", f"Confirmed via OpenAlex: '{best_title}'")
    if best_sim >= 0.5:
        return CitationResult("", title, VerifyStatus.SUSPICIOUS, best_sim, "openalex", f"Partial match via OpenAlex (sim={best_sim:.2f}): '{best_title}'")
    return CitationResult("", title, VerifyStatus.HALLUCINATED, 0.7, "openalex", "No close match found via OpenAlex")


def verify_by_title_search(title: str, *, s2_api_key: str = "") -> CitationResult | None:
    from paper_collect.literature.search import search_papers_multi_query

    try:
        results = search_papers_multi_query(
            [title],
            limit_per_query=5,
            sources=("semantic_scholar", "arxiv"),
            year_min=0,
            s2_api_key=s2_api_key,
            inter_query_delay=0.0,
        )
    except Exception:
        return None
    if not results.papers:
        return CitationResult("", title, VerifyStatus.HALLUCINATED, 0.7, "title_search", "No results found via Semantic Scholar + arXiv")
    best_sim = 0.0
    best_paper: Paper | None = None
    for paper in results.papers:
        sim = title_similarity(title, paper.title)
        if sim > best_sim:
            best_sim = sim
            best_paper = paper
    if best_sim >= 0.8 and best_paper is not None:
        return CitationResult("", title, VerifyStatus.VERIFIED, best_sim, "title_search", f"Found via search: '{best_paper.title}'", matched_paper=best_paper)
    if best_sim >= 0.5 and best_paper is not None:
        return CitationResult("", title, VerifyStatus.SUSPICIOUS, best_sim, "title_search", f"Partial match (sim={best_sim:.2f}): '{best_paper.title}'", matched_paper=best_paper)
    return CitationResult("", title, VerifyStatus.HALLUCINATED, 1.0 - best_sim, "title_search", f"Best match too weak (sim={best_sim:.2f}): '{best_paper.title}'" if best_paper else "No match found")


def verify_citations(
    bib_text: str,
    *,
    s2_api_key: str = "",
    inter_verify_delay: float = 1.5,
) -> VerificationReport:
    entries = parse_bibtex_entries(bib_text)
    report = VerificationReport(total=len(entries))
    delay = max(0.0, inter_verify_delay)
    for index, entry in enumerate(entries):
        if index > 0 and delay > 0:
            import time

            time.sleep(delay)
        result = _verify_entry(entry, s2_api_key=s2_api_key)
        report.results.append(result)
        if result.status == VerifyStatus.VERIFIED:
            report.verified += 1
        elif result.status == VerifyStatus.SUSPICIOUS:
            report.suspicious += 1
        elif result.status == VerifyStatus.HALLUCINATED:
            report.hallucinated += 1
        else:
            report.skipped += 1
    return report


def _verify_entry(entry: dict[str, str], *, s2_api_key: str) -> CitationResult:
    key = entry.get("key", "")
    title = entry.get("title", "").strip()
    doi = entry.get("doi", "").strip()
    arxiv_id = entry.get("eprint", "").strip() or entry.get("arxiv_id", "").strip()
    if not title:
        return CitationResult(key, "", VerifyStatus.SKIPPED, 0.0, "skipped", "No title in BibTeX entry")

    result: CitationResult | None = None
    if doi:
        result = verify_by_doi(doi, title)
    if result is None and arxiv_id:
        result = verify_by_arxiv_id(arxiv_id, title)
    if result is None:
        result = verify_by_openalex(title)
    if result is None:
        result = verify_by_title_search(title, s2_api_key=s2_api_key)
    if result is None:
        return CitationResult(key, title, VerifyStatus.SKIPPED, 0.0, "skipped", "All verification methods failed")
    return CitationResult(
        cite_key=key,
        title=title,
        status=result.status,
        confidence=result.confidence,
        method=result.method,
        details=result.details,
        matched_paper=result.matched_paper,
    )


def filter_verified_bibtex(
    bib_text: str,
    report: VerificationReport,
    *,
    include_suspicious: bool = True,
) -> str:
    entries = parse_bibtex_entries(bib_text)
    allowed_keys: set[str] = set()
    for result in report.results:
        if result.status == VerifyStatus.VERIFIED:
            allowed_keys.add(result.cite_key)
        elif include_suspicious and result.status in {VerifyStatus.SUSPICIOUS, VerifyStatus.SKIPPED}:
            allowed_keys.add(result.cite_key)
    rendered = []
    for entry in entries:
        if entry.get("key", "") not in allowed_keys:
            continue
        rendered.append(_entry_to_bibtex(entry))
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def _entry_to_bibtex(entry: dict[str, str]) -> str:
    entry_type = entry.get("type", "article")
    key = entry.get("key", "entry")
    lines = [f"@{entry_type}{{{key},"]
    for field in ("title", "author", "year", "journal", "booktitle", "doi", "eprint", "archiveprefix", "url"):
        value = entry.get(field)
        if value:
            lines.append(f"  {field} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def _xml_text(element: ET.Element, path: str) -> str:
    namespaces = {"atom": "http://www.w3.org/2005/Atom"}
    child = element.find(path, namespaces)
    return child.text.strip() if child is not None and child.text else ""
