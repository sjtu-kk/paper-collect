from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from paper_collect.literature import Author, Paper
from paper_collect.literature import arxiv_client, crossref_client, dblp_client, doaj_client, openalex_client, pubmed_client, search, semantic_scholar


class _Response:
    def __init__(self, payload: str | bytes) -> None:
        self._payload = payload.encode("utf-8") if isinstance(payload, str) else payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _paper(
    *,
    paper_id: str,
    title: str,
    citation_count: int,
    year: int,
    doi: str = "",
    arxiv_id: str = "",
    source: str = "openalex",
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=(Author(name="Ada Lovelace"),),
        year=year,
        abstract="abstract",
        venue="venue",
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        url=f"https://example.org/{paper_id}",
        source=source,
    )


def test_openalex_response_parses_into_paper(monkeypatch) -> None:  # noqa: ANN001
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Biomedical Agents",
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "publication_year": 2025,
                "primary_location": {"source": {"display_name": "Nature AI"}},
                "cited_by_count": 42,
                "doi": "https://doi.org/10.1000/agents",
                "ids": {"doi": "https://doi.org/10.1000/agents"},
                "abstract_inverted_index": {"Agent": [0], "systems": [1]},
            }
        ]
    }
    monkeypatch.setattr(openalex_client.urllib.request, "urlopen", lambda *args, **kwargs: _Response(json.dumps(payload)))

    papers = openalex_client.search_openalex("biomedical agents", limit=1, year_min=2020)

    assert len(papers) == 1
    assert papers[0].paper_id == "https://openalex.org/W1"
    assert papers[0].title == "Biomedical Agents"
    assert [author.name for author in papers[0].authors] == ["Ada Lovelace"]
    assert papers[0].year == 2025
    assert papers[0].venue == "Nature AI"
    assert papers[0].citation_count == 42
    assert papers[0].doi == "10.1000/agents"
    assert papers[0].abstract == "Agent systems"
    assert papers[0].source == "openalex"


def test_openalex_response_preserves_source_authority_metadata(monkeypatch) -> None:  # noqa: ANN001
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W2",
                "title": "Source Authority Agents",
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "publication_year": 2026,
                "primary_location": {
                    "source": {
                        "id": "https://openalex.org/S1983995261",
                        "display_name": "Nature Machine Intelligence",
                        "issn_l": "2522-5839",
                        "issn": ["2522-5839", "2522-5847"],
                        "type": "journal",
                        "host_organization": "https://openalex.org/P4310320990",
                    }
                },
                "cited_by_count": 7,
                "doi": "https://doi.org/10.1000/source-agents",
                "ids": {"openalex": "https://openalex.org/W2"},
                "abstract_inverted_index": {},
            }
        ]
    }
    monkeypatch.setattr(openalex_client.urllib.request, "urlopen", lambda *args, **kwargs: _Response(json.dumps(payload)))

    papers = openalex_client.search_openalex("source authority agents", limit=1, year_min=2020)

    assert papers[0].source_metadata == {
        "evidence_kind": "openalex_source_authority",
        "source_id": "https://openalex.org/S1983995261",
        "display_name": "Nature Machine Intelligence",
        "issn_l": "2522-5839",
        "issn": ["2522-5839", "2522-5847"],
        "type": "journal",
        "host_organization": "https://openalex.org/P4310320990",
    }


def test_openalex_source_authority_metadata_does_not_invent_missing_fields(monkeypatch) -> None:  # noqa: ANN001
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W3",
                "title": "Partial Source Authority Agents",
                "authorships": [],
                "publication_year": 2026,
                "primary_location": {"source": {"display_name": "Sparse Venue"}},
                "cited_by_count": 1,
                "doi": "",
                "ids": {},
                "abstract_inverted_index": {},
            }
        ]
    }
    monkeypatch.setattr(openalex_client.urllib.request, "urlopen", lambda *args, **kwargs: _Response(json.dumps(payload)))

    papers = openalex_client.search_openalex("partial source authority agents", limit=1, year_min=2020)

    assert papers[0].source_metadata == {
        "evidence_kind": "openalex_source_authority",
        "source_id": "",
        "display_name": "Sparse Venue",
        "issn_l": "",
        "issn": [],
        "type": "",
        "host_organization": "",
    }


def test_semantic_scholar_response_parses_into_paper(monkeypatch) -> None:  # noqa: ANN001
    payload = {
        "data": [
            {
                "paperId": "S2-1",
                "title": "Clinical Research Agents",
                "abstract": "agent abstract",
                "year": 2024,
                "venue": "ACL",
                "citationCount": 19,
                "authors": [{"name": "Grace Hopper"}],
                "externalIds": {"DOI": "doi:10.2000/s2", "ArXiv": "2401.00001"},
                "url": "https://semanticscholar.org/paper/S2-1",
            }
        ]
    }
    monkeypatch.setattr(semantic_scholar.urllib.request, "urlopen", lambda *args, **kwargs: _Response(json.dumps(payload)))

    papers = semantic_scholar.search_semantic_scholar("clinical agents", limit=1, year_min=2020, api_key="key")

    assert len(papers) == 1
    assert papers[0].paper_id == "S2-1"
    assert papers[0].title == "Clinical Research Agents"
    assert [author.name for author in papers[0].authors] == ["Grace Hopper"]
    assert papers[0].doi == "10.2000/s2"
    assert papers[0].arxiv_id == "2401.00001"
    assert papers[0].source == "semantic_scholar"


def test_arxiv_request_page_size_follows_task_limit(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}

    class _SortCriterion:
        Relevance = "relevance"
        SubmittedDate = "submitted"

    class _Search:
        def __init__(self, *, query, max_results, sort_by):  # noqa: ANN001
            captured["query"] = query
            captured["max_results"] = max_results
            captured["sort_by"] = sort_by

    class _Client:
        def __init__(self, *, page_size, delay_seconds, num_retries):  # noqa: ANN001
            captured["page_size"] = page_size
            captured["delay_seconds"] = delay_seconds
            captured["num_retries"] = num_retries

        def results(self, search_obj):  # noqa: ANN001
            captured["search_obj"] = search_obj
            return []

    fake_arxiv = SimpleNamespace(SortCriterion=_SortCriterion, Search=_Search, Client=_Client)
    monkeypatch.setattr(arxiv_client, "arxiv", fake_arxiv)

    papers = arxiv_client.search_arxiv("graph attention networks", limit=3, year_min=2020)

    assert papers == []
    assert captured["max_results"] == 3
    assert captured["page_size"] == 3
    assert captured["delay_seconds"] >= 3.0


def test_rate_limited_providers_expose_provider_local_backoff_policy(monkeypatch) -> None:  # noqa: ANN001
    def fake_search_arxiv(*args, **kwargs):  # noqa: ANN001, ANN202
        raise search.ProviderBlocked("arxiv", "rate_limited", "arXiv rate limited")

    def fake_search_semantic_scholar(*args, **kwargs):  # noqa: ANN001, ANN202
        raise search.ProviderBlocked("semantic_scholar", "rate_limited", "Semantic Scholar rate limited")

    monkeypatch.setattr(search, "search_arxiv", fake_search_arxiv, raising=False)
    monkeypatch.setattr(search, "search_semantic_scholar", fake_search_semantic_scholar, raising=False)

    arxiv_result = search._search_source("arxiv", "graph attention networks", limit=1, year_min=2020, s2_api_key="")
    s2_result = search._search_source("semantic_scholar", "graph attention networks", limit=1, year_min=2020, s2_api_key="")

    assert arxiv_result["status"]["rate_control"] == {
        "provider_local": True,
        "strategy": "throttle_backoff_circuit_breaker",
        "circuit_breaker": "open_for_provider",
    }
    assert s2_result["status"]["rate_control"] == {
        "provider_local": True,
        "strategy": "throttle_backoff_circuit_breaker",
        "circuit_breaker": "open_for_provider",
    }


def test_crossref_request_and_response_preserve_bibliographic_authority_metadata(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1145/1234567.8901234",
                    "title": ["Agentic Literature Search"],
                    "author": [{"given": "Ada", "family": "Lovelace"}, {"given": "Grace", "family": "Hopper"}],
                    "issued": {"date-parts": [[2024, 6, 1]]},
                    "container-title": ["Proceedings of the ACM Conference on Research Agents"],
                    "publisher": "Association for Computing Machinery",
                    "ISSN": ["1234-5678", "8765-4321"],
                    "URL": "https://doi.org/10.1145/1234567.8901234",
                    "type": "proceedings-article",
                    "is-referenced-by-count": 17,
                    "abstract": "<jats:p>Agentic search over literature.</jats:p>",
                }
            ]
        }
    }

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        return _Response(json.dumps(payload))

    monkeypatch.setattr(crossref_client.urllib.request, "urlopen", fake_urlopen)

    papers = crossref_client.search_crossref("agentic literature search", limit=2, year_min=2020)

    assert "api.crossref.org/works" in str(captured["url"])
    assert "query=agentic+literature+search" in str(captured["url"])
    assert "rows=2" in str(captured["url"])
    assert "from-pub-date%3A2020-01-01" in str(captured["url"])
    assert len(papers) == 1
    assert papers[0].paper_id == "10.1145/1234567.8901234"
    assert papers[0].title == "Agentic Literature Search"
    assert [author.name for author in papers[0].authors] == ["Ada Lovelace", "Grace Hopper"]
    assert papers[0].year == 2024
    assert papers[0].venue == "Proceedings of the ACM Conference on Research Agents"
    assert papers[0].citation_count == 17
    assert papers[0].doi == "10.1145/1234567.8901234"
    assert papers[0].url == "https://doi.org/10.1145/1234567.8901234"
    assert papers[0].source == "crossref"
    assert papers[0].source_metadata == {
        "evidence_kind": "crossref_bibliographic_authority",
        "provider_id": "crossref",
        "crossref_id": "10.1145/1234567.8901234",
        "container_title": "Proceedings of the ACM Conference on Research Agents",
        "issn": ["1234-5678", "8765-4321"],
        "publisher": "Association for Computing Machinery",
        "type": "proceedings-article",
    }


def test_crossref_rate_limit_returns_structured_provider_status(monkeypatch) -> None:  # noqa: ANN001
    def fake_search_crossref(*args, **kwargs):  # noqa: ANN001, ANN202
        raise search.ProviderBlocked("crossref", "rate_limited", "Crossref rate limited")

    monkeypatch.setattr(search, "search_crossref", fake_search_crossref, raising=False)

    result = search._search_source("crossref", "agentic literature search", limit=1, year_min=2020, s2_api_key="")

    assert result == {
        "papers": [],
        "status": {
            "source_name": "crossref",
            "status": "rate_limited",
            "returned_count": 0,
            "warnings": ["Crossref rate limited"],
        },
    }


def test_dblp_request_and_response_preserve_cs_bibliographic_metadata(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}
    payload = {
        "result": {
            "hits": {
                "hit": [
                    {
                        "info": {
                            "key": "conf/iclr/VelickovicCCRLB18",
                            "type": "Conference and Workshop Papers",
                            "authors": {
                                "author": [
                                    {"text": "Petar Velickovic"},
                                    {"text": "Guillem Cucurull"},
                                ]
                            },
                            "title": "Graph Attention Networks.",
                            "venue": "ICLR",
                            "year": "2018",
                            "doi": "10.48550/arXiv.1710.10903",
                            "ee": "https://doi.org/10.48550/arXiv.1710.10903",
                            "url": "https://dblp.org/rec/conf/iclr/VelickovicCCRLB18",
                        }
                    }
                ]
            }
        }
    }

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        return _Response(json.dumps(payload))

    monkeypatch.setattr(dblp_client.urllib.request, "urlopen", fake_urlopen)

    papers = dblp_client.search_dblp("graph attention networks", limit=2, year_min=2017)

    assert "dblp.org/search/publ/api" in str(captured["url"])
    assert "q=graph+attention+networks" in str(captured["url"])
    assert "h=2" in str(captured["url"])
    assert "format=json" in str(captured["url"])
    assert len(papers) == 1
    assert papers[0].paper_id == "conf/iclr/VelickovicCCRLB18"
    assert papers[0].title == "Graph Attention Networks."
    assert [author.name for author in papers[0].authors] == ["Petar Velickovic", "Guillem Cucurull"]
    assert papers[0].year == 2018
    assert papers[0].venue == "ICLR"
    assert papers[0].doi == "10.48550/arXiv.1710.10903"
    assert papers[0].url == "https://dblp.org/rec/conf/iclr/VelickovicCCRLB18"
    assert papers[0].abstract == ""
    assert papers[0].source == "dblp"
    assert papers[0].source_metadata == {
        "evidence_kind": "dblp_cs_bibliographic",
        "provider_id": "dblp",
        "dblp_key": "conf/iclr/VelickovicCCRLB18",
        "venue": "ICLR",
        "publication_type": "Conference and Workshop Papers",
        "ee": "https://doi.org/10.48550/arXiv.1710.10903",
    }


def test_dblp_compacts_long_provider_query_and_preserves_original(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}
    long_query = "graph attention networks for interpretable neural message passing with irrelevant provider prose and survey framing"
    payload = {"result": {"hits": {"hit": []}}}

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["url"] = request.full_url
        return _Response(json.dumps(payload))

    monkeypatch.setattr(dblp_client.urllib.request, "urlopen", fake_urlopen)

    papers = dblp_client.search_dblp(long_query, limit=2, year_min=2020)

    assert papers == []
    assert "q=graph+attention+networks+interpretable+neural+message+passing" in str(captured["url"])
    assert "irrelevant" not in str(captured["url"])

    result = search._search_source("dblp", long_query, limit=2, year_min=2020, s2_api_key="")

    assert result["status"]["status"] == "completed_no_results"
    assert result["status"]["returned_count"] == 0
    assert result["compiled_query_summary"] == {
        "original_query_text": long_query,
        "provider_query_text": "graph attention networks interpretable neural message passing",
        "rewrite_reason": "dblp_keyword_compaction",
    }


def test_dblp_network_error_returns_structured_provider_status(monkeypatch) -> None:  # noqa: ANN001
    def fake_search_dblp(*args, **kwargs):  # noqa: ANN001, ANN202
        raise search.ProviderBlocked("dblp", "network_failed", "DBLP network unavailable")

    monkeypatch.setattr(search, "search_dblp", fake_search_dblp, raising=False)

    result = search._search_source("dblp", "graph attention networks", limit=1, year_min=2020, s2_api_key="")

    assert result == {
        "papers": [],
        "status": {
            "source_name": "dblp",
            "status": "network_failed",
            "returned_count": 0,
            "warnings": ["DBLP network unavailable"],
        },
    }


def test_doaj_request_and_response_preserve_oa_journal_article_metadata(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}
    payload = {
        "results": [
            {
                "id": "doaj-article-1",
                "bibjson": {
                    "title": "Open access graph attention networks",
                    "year": "2024",
                    "author": [{"name": "Ada Author", "affiliation": "Open Lab"}],
                    "abstract": "OA article metadata",
                    "identifier": [
                        {"type": "doi", "id": "10.1234/doaj.1"},
                        {"type": "eissn", "id": "1234-5678"},
                    ],
                    "journal": {
                        "title": "Journal of Open Metadata",
                        "publisher": "Open Publisher",
                        "language": ["EN"],
                        "issns": ["1234-5678"],
                    },
                    "license": [{"title": "CC BY", "type": "CC BY"}],
                    "link": [{"type": "fulltext", "url": "https://example.org/article-1", "content_type": "HTML"}],
                },
            },
            {
                "id": "old-doaj-article",
                "bibjson": {
                    "title": "Old OA article",
                    "year": "2018",
                    "journal": {"title": "Old Journal"},
                },
            },
        ]
    }

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return _Response(json.dumps(payload))

    monkeypatch.setattr(doaj_client.urllib.request, "urlopen", fake_urlopen)

    papers = doaj_client.search_doaj("graph attention networks", limit=2, year_min=2020)

    assert "doaj.org/api/v4/search/articles" in str(captured["url"])
    assert "graph%20attention%20networks" in str(captured["url"])
    assert "pageSize=2" in str(captured["url"])
    assert len(papers) == 1
    assert papers[0].paper_id == "doaj-article-1"
    assert papers[0].title == "Open access graph attention networks"
    assert papers[0].authors[0].name == "Ada Author"
    assert papers[0].authors[0].affiliation == "Open Lab"
    assert papers[0].year == 2024
    assert papers[0].abstract == "OA article metadata"
    assert papers[0].venue == "Journal of Open Metadata"
    assert papers[0].doi == "10.1234/doaj.1"
    assert papers[0].source == "doaj"
    assert papers[0].source_metadata == {
        "evidence_kind": "doaj_oa_metadata",
        "provider_id": "doaj",
        "doaj_id": "doaj-article-1",
        "journal_title": "Journal of Open Metadata",
        "issn": ["1234-5678"],
        "publisher": "Open Publisher",
        "license": ["CC BY"],
        "language": ["EN"],
        "link_url": "https://example.org/article-1",
        "link_type": "fulltext",
    }


def test_doaj_network_error_returns_structured_provider_status(monkeypatch) -> None:  # noqa: ANN001
    def fake_search_doaj(*args, **kwargs):  # noqa: ANN001, ANN202
        raise search.ProviderBlocked("doaj", "timeout", "DOAJ request timed out")

    monkeypatch.setattr(search, "search_doaj", fake_search_doaj, raising=False)

    result = search._search_source("doaj", "open access metadata", limit=1, year_min=2020, s2_api_key="")

    assert result == {
        "papers": [],
        "status": {
            "source_name": "doaj",
            "status": "timeout",
            "returned_count": 0,
            "warnings": ["DOAJ request timed out"],
        },
    }


def test_pubmed_request_and_response_preserve_biomedical_metadata(monkeypatch) -> None:  # noqa: ANN001
    captured_urls: list[str] = []
    search_payload = {"esearchresult": {"idlist": ["42184189", "42174858"]}}
    summary_payload = {
        "result": {
            "uids": ["42184189", "42174858"],
            "42184189": {
                "uid": "42184189",
                "pubdate": "2026 May 25",
                "source": "IEEE J Biomed Health Inform",
                "fulljournalname": "IEEE journal of biomedical and health informatics",
                "title": "An Adaptive Fusion Network for Breast Tumor Grading Based on Graph Structure Learning.",
                "authors": [{"name": "Yang L"}, {"name": "Ma L"}],
                "articleids": [
                    {"idtype": "pubmed", "value": "42184189"},
                    {"idtype": "doi", "value": "10.1109/JBHI.2026.3696620"},
                ],
                "pubtype": ["Journal Article"],
                "lang": ["eng"],
                "attributes": ["Has Abstract"],
            },
            "42174858": {
                "uid": "42174858",
                "pubdate": "2018 May 21",
                "source": "Old Biomed J",
                "title": "Old biomedical graph article.",
            },
        }
    }

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        captured_urls.append(req.full_url)
        if "esearch.fcgi" in req.full_url:
            return _Response(json.dumps(search_payload))
        if "esummary.fcgi" in req.full_url:
            return _Response(json.dumps(summary_payload))
        raise AssertionError(req.full_url)

    monkeypatch.setattr(pubmed_client.urllib.request, "urlopen", fake_urlopen)

    papers = pubmed_client.search_pubmed("graph attention networks cancer", limit=2, year_min=2020)

    assert len(captured_urls) == 2
    assert "esearch.fcgi" in captured_urls[0]
    assert "graph+attention+networks+cancer" in captured_urls[0]
    assert "mindate=2020" in captured_urls[0]
    assert "esummary.fcgi" in captured_urls[1]
    assert len(papers) == 1
    assert papers[0].paper_id == "42184189"
    assert papers[0].title == "An Adaptive Fusion Network for Breast Tumor Grading Based on Graph Structure Learning."
    assert papers[0].authors[0].name == "Yang L"
    assert papers[0].year == 2026
    assert papers[0].abstract == "Has Abstract"
    assert papers[0].venue == "IEEE journal of biomedical and health informatics"
    assert papers[0].doi == "10.1109/JBHI.2026.3696620"
    assert papers[0].url == "https://pubmed.ncbi.nlm.nih.gov/42184189/"
    assert papers[0].source == "pubmed"
    assert papers[0].source_metadata == {
        "evidence_kind": "pubmed_biomedical_corpus",
        "provider_id": "pubmed",
        "pmid": "42184189",
        "journal": "IEEE journal of biomedical and health informatics",
        "source": "IEEE J Biomed Health Inform",
        "publication_types": ["Journal Article"],
        "language": ["eng"],
        "has_abstract": True,
    }


def test_pubmed_network_error_returns_structured_provider_status(monkeypatch) -> None:  # noqa: ANN001
    def fake_search_pubmed(*args, **kwargs):  # noqa: ANN001, ANN202
        raise search.ProviderBlocked("pubmed", "network_failed", "PubMed network unavailable")

    monkeypatch.setattr(search, "search_pubmed", fake_search_pubmed, raising=False)

    result = search._search_source("pubmed", "biomedical graph", limit=1, year_min=2020, s2_api_key="")

    assert result == {
        "papers": [],
        "status": {
            "source_name": "pubmed",
            "status": "network_failed",
            "returned_count": 0,
            "warnings": ["PubMed network unavailable"],
        },
    }


def test_arxiv_library_result_parses_into_paper(monkeypatch) -> None:  # noqa: ANN001
    result = SimpleNamespace(
        title="Agent Benchmarks",
        authors=[SimpleNamespace(name="Alan Turing")],
        entry_id="https://arxiv.org/abs/2501.01234v2",
        doi="",
        journal_ref="arXiv",
        published=datetime(2025, 1, 2, tzinfo=timezone.utc),
        summary="benchmark abstract",
    )

    class FakeClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.kwargs = kwargs

        def results(self, search_obj):  # noqa: ANN001
            assert search_obj.query == 'all:"agent benchmarks"'
            yield result

    fake_arxiv = SimpleNamespace(
        SortCriterion=SimpleNamespace(Relevance="relevance", SubmittedDate="submitted"),
        Search=lambda **kwargs: SimpleNamespace(**kwargs),
        Client=FakeClient,
    )
    monkeypatch.setattr(arxiv_client, "arxiv", fake_arxiv)

    papers = arxiv_client.search_arxiv("agent benchmarks", limit=1, year_min=2020)

    assert len(papers) == 1
    assert papers[0].paper_id == "https://arxiv.org/abs/2501.01234v2"
    assert papers[0].title == "Agent Benchmarks"
    assert papers[0].arxiv_id == "2501.01234"
    assert papers[0].url == "https://arxiv.org/abs/2501.01234v2"
    assert papers[0].source == "arxiv"


def test_search_deduplicates_by_doi_then_arxiv_then_normalized_title_and_sorts(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        search,
        "search_openalex",
        lambda *args, **kwargs: [
            _paper(paper_id="doi-low", title="Duplicate DOI", citation_count=1, year=2020, doi="10.1/dup", source="openalex"),
            _paper(paper_id="title-low", title="Same Title!", citation_count=3, year=2021, source="openalex"),
        ],
    )
    monkeypatch.setattr(
        search,
        "search_semantic_scholar",
        lambda *args, **kwargs: [
            _paper(paper_id="doi-high", title="Duplicate DOI Better", citation_count=20, year=2024, doi="10.1/DUP", source="semantic_scholar"),
            _paper(paper_id="arxiv-low", title="Duplicate arXiv", citation_count=2, year=2022, arxiv_id="2401.00001", source="semantic_scholar"),
        ],
    )
    monkeypatch.setattr(
        search,
        "search_arxiv",
        lambda *args, **kwargs: [
            _paper(paper_id="arxiv-high", title="Duplicate arXiv Better", citation_count=5, year=2025, arxiv_id="2401.00001", source="arxiv"),
            _paper(paper_id="title-high", title="Same Title", citation_count=10, year=2023, source="arxiv"),
        ],
    )
    monkeypatch.setattr(search, "search_crossref", lambda *args, **kwargs: [])
    monkeypatch.setattr(search, "search_dblp", lambda *args, **kwargs: [])
    monkeypatch.setattr(search, "search_doaj", lambda *args, **kwargs: [])
    monkeypatch.setattr(search, "search_pubmed", lambda *args, **kwargs: [])

    result = search.search_papers("agents", limit=5, year_min=2020)

    assert [paper.paper_id for paper in result.papers] == ["doi-high", "title-high", "arxiv-high"]
    assert result.status == "completed"
    assert [status["source_name"] for status in result.provider_statuses] == ["openalex", "semantic_scholar", "arxiv", "crossref", "dblp", "doaj", "pubmed"]
