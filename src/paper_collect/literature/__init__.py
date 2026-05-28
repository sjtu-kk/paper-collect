# Derived from AutoResearchClaw (MIT License). Copyright (c) 2026 Aiming Lab.

from __future__ import annotations

from paper_collect.literature.arxiv_client import search_arxiv
from paper_collect.literature.crossref_client import search_crossref
from paper_collect.literature.dblp_client import search_dblp
from paper_collect.literature.doaj_client import search_doaj
from paper_collect.literature.models import Author, Paper
from paper_collect.literature.openalex_client import search_openalex
from paper_collect.literature.pubmed_client import search_pubmed
from paper_collect.literature.search import SearchBatchResult, papers_to_bibtex, search_papers, search_papers_multi_query
from paper_collect.literature.semantic_scholar import search_semantic_scholar
from paper_collect.literature.verify import (
    CitationResult,
    VerificationReport,
    VerifyStatus,
    filter_verified_bibtex,
    parse_bibtex_entries,
    title_similarity,
    verify_citations,
)

__all__ = [
    "Author",
    "CitationResult",
    "Paper",
    "SearchBatchResult",
    "VerificationReport",
    "VerifyStatus",
    "filter_verified_bibtex",
    "papers_to_bibtex",
    "parse_bibtex_entries",
    "search_arxiv",
    "search_crossref",
    "search_dblp",
    "search_doaj",
    "search_openalex",
    "search_papers",
    "search_papers_multi_query",
    "search_pubmed",
    "search_semantic_scholar",
    "title_similarity",
    "verify_citations",
]
