# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.
"""Data models for literature search results."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Author:
    name: str
    affiliation: str = ""

    def last_name(self) -> str:
        parts = self.name.strip().split()
        raw = parts[-1] if parts else "unknown"
        nfkd = unicodedata.normalize("NFKD", raw)
        ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z]", "", ascii_name).lower() or "unknown"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "affiliation": self.affiliation}


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[Author, ...] = ()
    year: int = 0
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    _bibtex_override: str = field(default="", repr=False)

    @property
    def cite_key(self) -> str:
        last = self.authors[0].last_name() if self.authors else "anon"
        yr = str(self.year) if self.year else "0000"
        keyword = ""
        for word in self.title.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                keyword = cleaned
                break
        return f"{last}{yr}{keyword}"

    def to_bibtex(self) -> str:
        if self._bibtex_override:
            return self._bibtex_override.strip()

        key = self.cite_key
        authors_str = " and ".join(author.name for author in self.authors) or "Unknown"
        venue = self.venue or ""
        is_arxiv_category = bool(
            re.match(
                r"^(?:cs|math|stat|eess|physics|q-bio|q-fin|astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|nlin|nucl-ex|nucl-th|quant-ph)\.[A-Z]{2}$",
                venue,
            )
        )

        if venue and not is_arxiv_category and any(
            keyword in venue.lower()
            for keyword in ("conference", "proc", "workshop", "neurips", "icml", "iclr", "aaai", "cvpr", "acl", "emnlp", "naacl", "eccv", "iccv", "sigir", "kdd", "www", "ijcai")
        ):
            entry_type = "inproceedings"
            venue_field = f"  booktitle = {{{venue}}},"
        elif self.arxiv_id and (not venue or is_arxiv_category):
            entry_type = "article"
            venue_field = f"  journal = {{arXiv preprint arXiv:{self.arxiv_id}}},"
        else:
            entry_type = "article"
            venue_field = f"  journal = {{{venue or 'Unknown'}}}," if venue else ""

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{self.title}}},")
        lines.append(f"  author = {{{authors_str}}},")
        lines.append(f"  year = {{{self.year or 'Unknown'}}},")
        if venue_field:
            lines.append(venue_field)
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{self.arxiv_id}}},")
            lines.append("  archiveprefix = {arXiv},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [author.to_dict() for author in self.authors],
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "source": self.source,
            "source_metadata": dict(self.source_metadata),
            "cite_key": self.cite_key,
        }


_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "upon",
        "about",
        "through",
        "using",
        "based",
        "towards",
        "toward",
        "between",
        "under",
        "more",
        "than",
        "when",
        "what",
        "which",
        "where",
        "does",
        "have",
        "been",
        "some",
        "each",
        "also",
        "much",
        "very",
        "learning",
    }
)
