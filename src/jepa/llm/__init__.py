"""LLM commentary backends and services for JEPA demos."""

from .backends import (
    BackendStatus,
    ColabAICommentaryBackend,
    CommentaryBackend,
    DeterministicCommentaryBackend,
    GeminiCommentaryBackend,
    OpenAIResponsesCommentaryBackend,
    build_default_commentary_backend,
)
from .schema import (
    COMMENTARY_JSON_SCHEMA,
    CommentarySchemaError,
    ValidatedCommentaryPayload,
    validate_commentary_payload,
)
from .service import (
    CommentaryGenerationResult,
    LLMCommentaryService,
    build_default_commentary_service,
    generate_commentary,
)

__all__ = [
    "BackendStatus",
    "COMMENTARY_JSON_SCHEMA",
    "ColabAICommentaryBackend",
    "CommentaryBackend",
    "CommentaryGenerationResult",
    "CommentarySchemaError",
    "DeterministicCommentaryBackend",
    "GeminiCommentaryBackend",
    "LLMCommentaryService",
    "OpenAIResponsesCommentaryBackend",
    "ValidatedCommentaryPayload",
    "build_default_commentary_backend",
    "build_default_commentary_service",
    "generate_commentary",
    "validate_commentary_payload",
]
