from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

VALID_REGISTRY_STATES = frozenset({"active", "reference", "deferred", "unsupported", "needs_live_probe"})
VALID_ACCESS_TAGS = frozenset(
    {
        "public",
        "api_key_required",
        "email_required",
        "token_recommended",
        "paid_or_institutional",
        "network_risk",
        "rate_limited",
        "unknown",
        "restricted",
        "manual_only",
    }
)
VALID_VENUE_COVERAGE_KINDS = frozenset(
    {
        "entity",
        "searchable_filter",
        "string_field",
        "nested_field",
        "returned_only",
        "inferred",
        "none",
        "high_friction",
        "needs_live_probe",
    }
)


@dataclass(frozen=True)
class SupportedFilters:
    implemented: tuple[str, ...] = ()
    documented_reference: tuple[str, ...] = ()
    future_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class AccessModel:
    tags: tuple[str, ...]


@dataclass(frozen=True)
class VenueCoverage:
    kind: str
    searchable: bool
    corpus_limited: bool
    examples: tuple[str, ...] = ()
    confidence: str = "research"
    notes: str = ""


@dataclass(frozen=True)
class ProviderRegistryEntry:
    provider_id: str
    display_name: str
    registry_state: str
    roles: tuple[str, ...]
    coverage_scope: tuple[str, ...]
    venue_coverage: VenueCoverage
    supported_search_surfaces: tuple[str, ...]
    supported_filters: SupportedFilters
    result_fields: tuple[str, ...]
    provenance_capabilities: tuple[str, ...]
    stage_usage: tuple[str, ...]
    access_model: AccessModel
    evidence_refs: tuple[str, ...]


@lru_cache(maxsize=1)
def provider_registry_entries() -> tuple[ProviderRegistryEntry, ...]:
    entries = (
        ProviderRegistryEntry(
            provider_id="openalex",
            display_name="OpenAlex",
            registry_state="active",
            roles=("broad_metadata_discovery", "venue_authority", "cross_domain_discovery"),
            coverage_scope=("all_disciplines", "journals", "conferences", "repositories"),
            venue_coverage=VenueCoverage(
                kind="entity",
                searchable=True,
                corpus_limited=False,
                examples=("ACL", "CVPR", "AAAI", "Nature", "PNAS"),
                notes="First-class source entity via sources; works link back through primary_location.source.",
            ),
            supported_search_surfaces=("works.search", "title.search", "primary_location.source"),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("venue", "doi", "open_access", "publisher"),
                future_candidates=("category", "language", "date_range"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "abstract", "cited_by_count", "ids", "source_id", "source_display_name", "source_issn", "source_type"),
            provenance_capabilities=("returned_identifier", "returned_venue", "citation_signal", "source_authority"),
            stage_usage=("stage_03", "stage_04", "stage_23", "source_authority_visibility"),
            access_model=AccessModel(tags=("public", "email_required", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="semantic_scholar",
            display_name="Semantic Scholar",
            registry_state="active",
            roles=("metadata_enrichment", "citation_graph", "title_fallback"),
            coverage_scope=("all_disciplines", "citation_graph_coverage"),
            venue_coverage=VenueCoverage(
                kind="searchable_filter",
                searchable=True,
                corpus_limited=False,
                examples=("Nature", "Radiology", "N. Engl. J. Med."),
                notes="Search/filter and returned structured venue metadata; not a first-class venue registry.",
            ),
            supported_search_surfaces=("paper.search", "title.search"),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("venue", "citation_count", "fields_of_study"),
                future_candidates=("language", "date_range", "open_access"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "arxiv_id", "citation_count"),
            provenance_capabilities=("returned_identifier", "returned_venue", "citation_signal", "graph_enrichment"),
            stage_usage=("stage_03", "stage_04", "stage_23"),
            access_model=AccessModel(tags=("api_key_required", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="arxiv",
            display_name="arXiv",
            registry_state="active",
            roles=("preprint_discovery", "domain_corpus", "version_evidence"),
            coverage_scope=("preprint_server", "cs", "physics", "math", "biology"),
            venue_coverage=VenueCoverage(
                kind="string_field",
                searchable=True,
                corpus_limited=True,
                examples=("cs.CL", "Nature Communications", "Phys.Rev. D"),
                notes="Preprint corpus with category search and journal_ref text, not normalized venue registry.",
            ),
            supported_search_surfaces=("query_all", "id_list"),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("category", "author", "title"),
                future_candidates=("date_range", "sort_by"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "arxiv_id", "url"),
            provenance_capabilities=("returned_identifier", "returned_venue", "version_evidence"),
            stage_usage=("stage_03", "stage_04", "stage_23"),
            access_model=AccessModel(tags=("public", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="crossref",
            display_name="Crossref",
            registry_state="active",
            roles=("bibliographic_authority", "doi_resolution", "venue_authority"),
            coverage_scope=("all_disciplines", "journals", "conferences"),
            venue_coverage=VenueCoverage(
                kind="entity",
                searchable=True,
                corpus_limited=False,
                examples=("Nature", "CVPR 2011", "ICML '04", "HLT/EMNLP", "SIGIR '94", "ICSE 2001"),
                notes="DOI, journal, container-title, ISSN, publisher and proceedings authority metadata via works search.",
            ),
            supported_search_surfaces=("works.search",),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("doi", "issn", "publisher", "type"),
                future_candidates=("container_title",),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "url", "crossref_id", "container_title", "publisher", "issn", "type"),
            provenance_capabilities=("bibliographic_authority", "returned_identifier", "returned_venue"),
            stage_usage=("stage_03", "stage_04", "authority_visibility"),
            access_model=AccessModel(tags=("public", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="dblp",
            display_name="DBLP",
            registry_state="active",
            roles=("cs_bibliographic_authority", "venue_authority"),
            coverage_scope=("computer_science",),
            venue_coverage=VenueCoverage(
                kind="entity",
                searchable=True,
                corpus_limited=True,
                examples=("JMLR", "NeurIPS", "CVPR", "ACL", "EMNLP"),
                notes="CS venue and bibliography coverage only.",
            ),
            supported_search_surfaces=("publication.search",),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("year", "venue", "type"),
                future_candidates=("author", "doi"),
            ),
            result_fields=("title", "authors", "year", "venue", "url", "doi", "dblp_key", "publication_type", "ee"),
            provenance_capabilities=("cs_bibliographic_authority", "returned_identifier", "returned_venue"),
            stage_usage=("stage_03", "stage_04", "cs_bibliography_visibility"),
            access_model=AccessModel(tags=("public", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="doaj",
            display_name="DOAJ",
            registry_state="active",
            roles=("oa_journal_discovery", "oa_article_metadata", "venue_authority"),
            coverage_scope=("open_access_journals", "open_access_articles"),
            venue_coverage=VenueCoverage(
                kind="entity",
                searchable=True,
                corpus_limited=True,
                examples=("BJS Open", "Payesh", "Communications Engineering", "eLife"),
                notes="OA journal subset; public search separate from premium metadata services.",
            ),
            supported_search_surfaces=("articles.search",),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("subject", "country", "issn", "publisher", "language"),
                future_candidates=("date_range", "oa_only"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "url", "doaj_id", "journal_title", "issn", "publisher", "license", "language"),
            provenance_capabilities=("oa_metadata", "returned_identifier", "returned_venue"),
            stage_usage=("stage_03", "stage_04", "oa_metadata_visibility"),
            access_model=AccessModel(tags=("public", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="pubmed",
            display_name="PubMed",
            registry_state="active",
            roles=("biomedical_corpus", "bibliographic_authority"),
            coverage_scope=("biomedical_literature",),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=True,
                corpus_limited=True,
                examples=("Nature", "Nature Medicine", "JAMA", "Bioinformatics"),
                notes="Venue evidence is corpus-limited to biomedical records.",
            ),
            supported_search_surfaces=("esearch", "esummary"),
            supported_filters=SupportedFilters(
                implemented=("year_min",),
                documented_reference=("pub_year", "article_type", "mesh", "journal", "language"),
                future_candidates=("author",),
            ),
            result_fields=("title", "authors", "year", "venue", "pmid", "doi", "journal", "publication_types", "language", "has_abstract"),
            provenance_capabilities=("biomedical_corpus", "returned_identifier", "returned_venue"),
            stage_usage=("stage_03", "stage_04", "biomedical_corpus_visibility"),
            access_model=AccessModel(tags=("public", "email_required", "token_recommended", "rate_limited", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="pmc",
            display_name="PubMed Central",
            registry_state="reference",
            roles=("biomedical_full_text_corpus", "repository"),
            coverage_scope=("biomedical_full_text",),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=True,
                corpus_limited=True,
                examples=("Nature", "PLOS ONE", "eLife"),
                notes="Biomedical full-text corpus; not a general venue registry.",
            ),
            supported_search_surfaces=("search",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("journal", "author", "year"),
                future_candidates=("open_access",),
            ),
            result_fields=("title", "authors", "year", "venue", "pmcid", "doi"),
            provenance_capabilities=("returned_identifier", "returned_venue", "full_text_pointer"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "email_required", "token_recommended", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="europe_pmc",
            display_name="Europe PMC",
            registry_state="reference",
            roles=("biomedical_corpus", "oa_access"),
            coverage_scope=("biomedical_literature",),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=True,
                corpus_limited=True,
                examples=("Nature", "PLOS Medicine", "Genome Biology"),
                notes="Biomedical/life-science corpus-limited venue evidence.",
            ),
            supported_search_surfaces=("search",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("pub_year", "author", "journal"),
                future_candidates=("open_access",),
            ),
            result_fields=("title", "authors", "year", "venue", "pmcid", "doi"),
            provenance_capabilities=("returned_identifier", "returned_venue"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "unknown", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="unpaywall",
            display_name="Unpaywall",
            registry_state="reference",
            roles=("oa_access", "resolver"),
            coverage_scope=("oa_access_layer",),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=False,
                corpus_limited=True,
                examples=("Nature", "BioRxiv", "New England Journal of Medicine"),
                notes="OA resolver over DOI records; not venue discovery.",
            ),
            supported_search_surfaces=("doi_lookup",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("doi", "is_oa"),
                future_candidates=("license", "publisher"),
            ),
            result_fields=("doi", "is_oa", "best_oa_location", "host_type"),
            provenance_capabilities=("resolver_signal", "returned_identifier"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("email_required", "paid_or_institutional", "network_risk")),
            evidence_refs=("docs/research/paper-collect-venue-coverage-registry-decoupling.md",),
        ),
        ProviderRegistryEntry(
            provider_id="datacite",
            display_name="DataCite",
            registry_state="reference",
            roles=("pid_authority", "repository_metadata"),
            coverage_scope=("repository_metadata", "pid_graph"),
            venue_coverage=VenueCoverage(
                kind="nested_field",
                searchable=False,
                corpus_limited=True,
                examples=("Stimmen der Zeit", "CVPR 2007", "SIGGRAPH 2019", "NeurIPS"),
                notes="PID metadata, not first-class venue registry.",
            ),
            supported_search_surfaces=("metadata.search", "doi_lookup"),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("doi", "resource_type", "publisher"),
                future_candidates=("year", "journal"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "resource_type"),
            provenance_capabilities=("returned_identifier", "returned_venue", "pid_resolution"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "api_key_required", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="opencitations",
            display_name="OpenCitations",
            registry_state="deferred",
            roles=("citation_graph", "resolver"),
            coverage_scope=("citation_graph",),
            venue_coverage=VenueCoverage(
                kind="none",
                searchable=False,
                corpus_limited=True,
                notes="Citation graph/resolver role, not venue coverage.",
            ),
            supported_search_surfaces=("citation.search",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("doi",),
                future_candidates=("citation_count", "year"),
            ),
            result_fields=("doi", "citing", "cited", "citation_count"),
            provenance_capabilities=("citation_graph_signal", "returned_identifier"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="openaire",
            display_name="OpenAIRE",
            registry_state="deferred",
            roles=("oa_discovery", "repository", "citation_graph"),
            coverage_scope=("oa_corpus", "repository_metadata"),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=False,
                corpus_limited=True,
                examples=("Scientometrics", "Research Policy"),
                notes="Mainly returned-only venue strings in repository metadata.",
            ),
            supported_search_surfaces=("search",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("year", "doi"),
                future_candidates=("venue", "subject"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi"),
            provenance_capabilities=("returned_identifier", "returned_venue"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "network_risk", "unknown")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="core",
            display_name="CORE",
            registry_state="deferred",
            roles=("oa_repository", "full_text_corpus"),
            coverage_scope=("oa_full_text", "repository_metadata"),
            venue_coverage=VenueCoverage(
                kind="needs_live_probe",
                searchable=True,
                corpus_limited=True,
                examples=("Icarus", "Library Hi Tech", "Nature Scientific Data"),
                notes="Docs-backed journal fields need current live schema confirmation before active use.",
            ),
            supported_search_surfaces=("search",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("year", "doi"),
                future_candidates=("repository", "license"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "full_text_url"),
            provenance_capabilities=("returned_identifier", "full_text_pointer"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("api_key_required", "paid_or_institutional", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="hal",
            display_name="HAL",
            registry_state="deferred",
            roles=("repository", "oa_discovery"),
            coverage_scope=("repository_metadata", "multidisciplinary"),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=True,
                corpus_limited=True,
                examples=("Nature", "Scientometrics", "ORASIS"),
                notes="Repository-side journal/conference fields.",
            ),
            supported_search_surfaces=("search",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("year", "author", "doi"),
                future_candidates=("collection", "open_access"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi"),
            provenance_capabilities=("returned_identifier", "returned_venue"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="zenodo",
            display_name="Zenodo",
            registry_state="deferred",
            roles=("repository", "pid_authority"),
            coverage_scope=("repository_metadata", "dataset_and_paper_records"),
            venue_coverage=VenueCoverage(
                kind="returned_only",
                searchable=False,
                corpus_limited=True,
                examples=("Scientometrics", "Research Policy", "PLOS Computational Biology", "MIS Quarterly"),
                notes="Repository/PID metadata, not venue discovery.",
            ),
            supported_search_surfaces=("search", "doi_lookup"),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=("doi", "resource_type"),
                future_candidates=("year", "community"),
            ),
            result_fields=("title", "authors", "year", "venue", "doi", "url"),
            provenance_capabilities=("returned_identifier", "pid_resolution"),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("public", "token_recommended", "network_risk")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
        ProviderRegistryEntry(
            provider_id="pid_repository_citation_graph_family",
            display_name="PID / Repository / Citation Graph Family",
            registry_state="needs_live_probe",
            roles=("candidate_family", "planning_placeholder"),
            coverage_scope=("needs_live_probe",),
            venue_coverage=VenueCoverage(
                kind="needs_live_probe",
                searchable=False,
                corpus_limited=True,
                notes="Planning placeholder for provider-specific PID/repository/citation graph rows.",
            ),
            supported_search_surfaces=("needs_live_probe",),
            supported_filters=SupportedFilters(
                implemented=(),
                documented_reference=(),
                future_candidates=("doi", "repository", "citation_graph", "venue"),
            ),
            result_fields=(),
            provenance_capabilities=("needs_live_probe",),
            stage_usage=("phase_2_planning",),
            access_model=AccessModel(tags=("network_risk", "unknown")),
            evidence_refs=("docs/research/paper-collect-provider-blackbox-boundary-registry.md",),
        ),
    )
    validate_provider_registry(entries)
    return entries


def validate_provider_registry(entries: tuple[ProviderRegistryEntry, ...]) -> None:
    provider_ids: set[str] = set()
    for entry in entries:
        if entry.provider_id in provider_ids:
            raise ValueError(f"duplicate provider_id: {entry.provider_id}")
        provider_ids.add(entry.provider_id)
        validate_provider_registry_entry(entry)


def validate_provider_registry_entry(entry: ProviderRegistryEntry) -> None:
    if entry.registry_state not in VALID_REGISTRY_STATES:
        raise ValueError(f"unsupported registry_state for {entry.provider_id}: {entry.registry_state}")
    required_sequences = {
        "roles": entry.roles,
        "coverage_scope": entry.coverage_scope,
        "supported_search_surfaces": entry.supported_search_surfaces,
        "provenance_capabilities": entry.provenance_capabilities,
        "stage_usage": entry.stage_usage,
        "evidence_refs": entry.evidence_refs,
    }
    for field_name, values in required_sequences.items():
        if not values:
            raise ValueError(f"{field_name} is required for {entry.provider_id}")
    filters = entry.supported_filters
    if not isinstance(filters.implemented, tuple) or not isinstance(filters.documented_reference, tuple) or not isinstance(filters.future_candidates, tuple):
        raise ValueError(f"supported_filters must use tuple sections for {entry.provider_id}")
    if entry.registry_state == "active" and "year_min" not in filters.implemented:
        raise ValueError(f"active provider {entry.provider_id} must implement year_min")
    if entry.registry_state == "active" and not entry.result_fields:
        raise ValueError(f"result_fields is required for active provider {entry.provider_id}")
    if entry.venue_coverage.kind not in VALID_VENUE_COVERAGE_KINDS:
        raise ValueError(f"venue_coverage.kind is unsupported for {entry.provider_id}: {entry.venue_coverage.kind}")
    if not entry.access_model.tags:
        raise ValueError(f"access_model.tags is required for {entry.provider_id}")
    invalid_access_tags = set(entry.access_model.tags) - VALID_ACCESS_TAGS
    if invalid_access_tags:
        raise ValueError(f"access_model contains unsupported tags for {entry.provider_id}: {sorted(invalid_access_tags)}")


def provider_registry_entry_payload(entry: ProviderRegistryEntry) -> dict[str, Any]:
    return {
        "provider_id": entry.provider_id,
        "display_name": entry.display_name,
        "registry_state": entry.registry_state,
        "roles": list(entry.roles),
        "coverage_scope": list(entry.coverage_scope),
        "venue_coverage": {
            "kind": entry.venue_coverage.kind,
            "searchable": entry.venue_coverage.searchable,
            "corpus_limited": entry.venue_coverage.corpus_limited,
            "examples": list(entry.venue_coverage.examples),
            "confidence": entry.venue_coverage.confidence,
            "notes": entry.venue_coverage.notes,
        },
        "supported_search_surfaces": list(entry.supported_search_surfaces),
        "supported_filters": {
            "implemented": list(entry.supported_filters.implemented),
            "documented_reference": list(entry.supported_filters.documented_reference),
            "future_candidates": list(entry.supported_filters.future_candidates),
        },
        "result_fields": list(entry.result_fields),
        "provenance_capabilities": list(entry.provenance_capabilities),
        "stage_usage": list(entry.stage_usage),
        "access_model": {
            "tags": list(entry.access_model.tags),
        },
        "evidence_refs": list(entry.evidence_refs),
    }


def provider_registry_snapshot(*, generated_at: str) -> dict[str, Any]:
    entries = provider_registry_entries()
    return {
        "artifact_version": "paper_collect_provider_registry_snapshot.v1",
        "generated_at": generated_at,
        "active_provider_ids": list(active_provider_ids()),
        "provider_registry_entries": [provider_registry_entry_payload(entry) for entry in entries],
    }


def provider_entry(provider_id: str) -> ProviderRegistryEntry:
    normalized = provider_id.strip().casefold()
    for entry in provider_registry_entries():
        if entry.provider_id.casefold() == normalized:
            return entry
    raise KeyError(provider_id)


def active_provider_ids() -> tuple[str, ...]:
    return tuple(entry.provider_id for entry in provider_registry_entries() if entry.registry_state == "active")
