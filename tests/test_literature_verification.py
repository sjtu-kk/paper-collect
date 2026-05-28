from __future__ import annotations

import json
import urllib.error

from paper_collect.literature import Author, Paper
from paper_collect.literature.search import SearchBatchResult
from paper_collect.literature import verify
from paper_collect.literature.verify import CitationResult, VerificationReport, VerifyStatus


class _Response:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_parse_bibtex_entries_parses_multiple_entries() -> None:
    entries = verify.parse_bibtex_entries(
        """
@article{alpha,
  title = {Alpha Paper},
  author = {A. Author},
  year = {2025},
  doi = {10.1000/alpha},
}

@inproceedings{beta,
  title = {Beta Paper},
  booktitle = {ICLR},
  year = {2024},
  eprint = {2401.00001},
}
"""
    )

    assert [entry["key"] for entry in entries] == ["alpha", "beta"]
    assert entries[0]["doi"] == "10.1000/alpha"
    assert entries[1]["eprint"] == "2401.00001"


def test_verify_by_doi_uses_datacite_fallback_after_crossref_404(monkeypatch) -> None:  # noqa: ANN001
    calls: list[str] = []

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        url = req.full_url
        calls.append(url)
        if "api.crossref.org" in url:
            raise urllib.error.HTTPError(url, 404, "not found", hdrs=None, fp=None)
        assert "api.datacite.org" in url
        return _Response(
            json.dumps(
                {
                    "data": {
                        "attributes": {
                            "titles": [{"title": "Alpha Paper"}],
                        }
                    }
                }
            )
        )

    monkeypatch.setattr(verify.urllib.request, "urlopen", fake_urlopen)

    result = verify.verify_by_doi("10.48550/arXiv.2401.00001", "Alpha Paper")

    assert result is not None
    assert result.status == VerifyStatus.VERIFIED
    assert result.method == "doi"
    assert "DataCite" in result.details
    assert [url.split("/")[2] for url in calls] == ["api.crossref.org", "api.datacite.org"]


def test_verify_citations_uses_title_search_fallback(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(verify, "verify_by_doi", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "verify_by_arxiv_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "verify_by_openalex", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        verify,
        "verify_by_title_search",
        lambda title, **kwargs: CitationResult("", title, VerifyStatus.VERIFIED, 0.91, "title_search", "found"),
    )

    report = verify.verify_citations(
        "@article{alpha,\n  title = {Alpha Paper},\n  year = {2025},\n}\n",
        inter_verify_delay=0.0,
    )

    assert report.total == 1
    assert report.verified == 1
    assert report.results[0].method == "title_search"


def test_title_search_fallback_uses_semantic_scholar_and_arxiv_only(monkeypatch) -> None:  # noqa: ANN001
    captured = {}

    def fake_search_papers_multi_query(queries, **kwargs):  # noqa: ANN001, ANN202
        captured["queries"] = list(queries)
        captured["kwargs"] = dict(kwargs)
        return SearchBatchResult(
            queries_used=list(queries),
            papers=[
                Paper(
                    paper_id="s2-1",
                    title="Alpha Paper",
                    authors=(Author(name="A. Author"),),
                    year=2025,
                    source="semantic_scholar",
                )
            ],
            provider_statuses=[{"source_name": "semantic_scholar", "status": "completed"}],
            status="completed",
        )

    monkeypatch.setattr("paper_collect.literature.search.search_papers_multi_query", fake_search_papers_multi_query)

    result = verify.verify_by_title_search("Alpha Paper")

    assert result is not None
    assert result.status == VerifyStatus.VERIFIED
    assert captured["queries"] == ["Alpha Paper"]
    assert captured["kwargs"]["sources"] == ("semantic_scholar", "arxiv")


def test_filter_verified_bibtex_retains_verified_suspicious_and_skipped_only() -> None:
    bib_text = """
@article{verified,
  title = {Verified Paper},
  year = {2025},
}

@article{suspicious,
  title = {Suspicious Paper},
  year = {2024},
}

@article{skipped,
  title = {Skipped Paper},
  year = {2023},
}

@article{hallucinated,
  title = {Missing Paper},
  year = {2022},
}
"""
    report = VerificationReport(
        total=4,
        verified=1,
        suspicious=1,
        hallucinated=1,
        skipped=1,
        results=[
            CitationResult("verified", "Verified Paper", VerifyStatus.VERIFIED, 1.0, "doi"),
            CitationResult("suspicious", "Suspicious Paper", VerifyStatus.SUSPICIOUS, 0.6, "openalex"),
            CitationResult("skipped", "Skipped Paper", VerifyStatus.SKIPPED, 0.0, "skipped"),
            CitationResult("hallucinated", "Missing Paper", VerifyStatus.HALLUCINATED, 0.9, "title_search"),
        ],
    )

    filtered = verify.filter_verified_bibtex(bib_text, report, include_suspicious=True)

    assert "@article{verified" in filtered
    assert "@article{suspicious" in filtered
    assert "@article{skipped" in filtered
    assert "@article{hallucinated" not in filtered
