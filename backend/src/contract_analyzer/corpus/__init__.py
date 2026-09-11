"""Frozen legal corpus construction and retrieval."""

from contract_analyzer.corpus.eli import (
    act_identifier,
    act_url,
    eli_act_url,
    validate_act_metadata,
    validate_amendments_not_carried,
)
from contract_analyzer.corpus.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROBE_TEXT,
    EMBEDDING_QUERY_INSTRUCTION,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_REQUEST_MAX_INPUTS,
    EMBEDDINGS_API_KEY_ENV,
    EMBEDDINGS_PROVIDER,
    _embed,
    checked_unit_vectors,
    effective_embedding_provider,
    embed_query_text,
    embedding_credential_configured,
    embedding_space_probe,
    embedding_spaces_agree,
)
from contract_analyzer.corpus.errors import (
    CorpusBuildError,
    CorpusIntegrityError,
    EmbeddingError,
)
from contract_analyzer.corpus.force import (
    _is_editorial_repeal_placeholder,
    act_force_state,
    provision_force_state,
)
from contract_analyzer.corpus.html import (
    html_to_text,
    is_childless_stub_of,
    node_tree_fingerprint,
    path_to_url,
)
from contract_analyzer.corpus.identity import (
    CORPUS_ALIAS,
    CORPUS_SNAPSHOTS_COLLECTION,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    corpus_collection_name,
    corpus_snapshot_id,
    embedding_space_fingerprint,
    sha256_bytes,
    snapshot_record_id,
    unit_digest_of,
    validate_expected_hash,
)
from contract_analyzer.corpus.manifest import (
    load_manifest,
    repeated_articles_in,
    validate_repeated_articles,
)
from contract_analyzer.corpus.models import (
    ActCurrency,
    CorpusManifest,
    CorpusSnapshot,
    LegalUnit,
    ManifestAct,
    RepeatedArticleCount,
)
from contract_analyzer.corpus.qdrant_index import (
    QdrantCorpusIndex,
)
from contract_analyzer.corpus.retrieval import (
    _fusion_order,
)
from contract_analyzer.corpus.sparse import (
    sparse_document_vector,
    sparse_query_vector,
)
from contract_analyzer.domain import (
    ForceScope,
    ForceState,
    ForceValue,
)

__all__ = [
    "ActCurrency",
    "CORPUS_ALIAS",
    "CORPUS_SNAPSHOTS_COLLECTION",
    "CorpusBuildError",
    "CorpusIntegrityError",
    "CorpusManifest",
    "CorpusSnapshot",
    "DENSE_VECTOR_NAME",
    "EMBEDDINGS_API_KEY_ENV",
    "EMBEDDINGS_PROVIDER",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "EMBEDDING_REQUEST_MAX_INPUTS",
    "EMBEDDING_QUERY_INSTRUCTION",
    "EMBEDDING_QUERY_PREFIX",
    "effective_embedding_provider",
    "embed_query_text",
    "checked_unit_vectors",
    "embedding_credential_configured",
    "embedding_space_probe",
    "embedding_spaces_agree",
    "EMBEDDING_PROBE_TEXT",
    "EmbeddingError",
    "ForceScope",
    "ForceState",
    "ForceValue",
    "LegalUnit",
    "ManifestAct",
    "QdrantCorpusIndex",
    "SPARSE_VECTOR_NAME",
    "_embed",
    "_fusion_order",
    "_is_editorial_repeal_placeholder",
    "act_force_state",
    "act_identifier",
    "act_url",
    "corpus_collection_name",
    "corpus_snapshot_id",
    "embedding_space_fingerprint",
    "eli_act_url",
    "html_to_text",
    "is_childless_stub_of",
    "node_tree_fingerprint",
    "load_manifest",
    "RepeatedArticleCount",
    "repeated_articles_in",
    "validate_repeated_articles",
    "path_to_url",
    "provision_force_state",
    "sha256_bytes",
    "snapshot_record_id",
    "sparse_document_vector",
    "sparse_query_vector",
    "unit_digest_of",
    "validate_act_metadata",
    "validate_amendments_not_carried",
    "validate_expected_hash",
]
