"""Transport-neutral model-call contract, attempt policy and telemetry.

The request payload, the retry policy, the attempt record and the response
taxonomy live here rather than in :mod:`contract_analyzer.openai_compatible`,
which owns only the call itself. The separation is what keeps the pipeline
independent of the client library: the pipeline imports this module, never the
provider client, and a change of endpoint or of library cannot quietly change
what a recorded attempt means.
"""

from contract_analyzer.domain import AttemptStatus as AttemptStatus

from .attempts import (
    RETRYABLE_SERVER_STATUSES as RETRYABLE_SERVER_STATUSES,
)
from .attempts import (
    AttemptFault as AttemptFault,
)
from .attempts import (
    AttemptSuccess as AttemptSuccess,
)
from .attempts import (
    AttemptTelemetry as AttemptTelemetry,
)
from .attempts import (
    InvalidModelResponse as InvalidModelResponse,
)
from .attempts import (
    ModelCallFailed as ModelCallFailed,
)
from .attempts import (
    ModelClientError as ModelClientError,
)
from .attempts import (
    ModelDeadlineExceeded as ModelDeadlineExceeded,
)
from .attempts import (
    ModelIdentityChanged as ModelIdentityChanged,
)
from .attempts import (
    ToolTurn as ToolTurn,
)
from .attempts import (
    invalid_response as invalid_response,
)
from .attempts import (
    run_attempts as run_attempts,
)
from .parsing import (
    hash_json as hash_json,
)
from .parsing import (
    token_counts as token_counts,
)
from .parsing import (
    tool_turn_hash as tool_turn_hash,
)
from .parsing import (
    usage_from_body as usage_from_body,
)
from .payloads import (
    CONVERSE_RESERVED as CONVERSE_RESERVED,
)
from .payloads import (
    tool_payload as tool_payload,
)
from .payloads import (
    validate_request as validate_request,
)
from .protocols import ModelClient as ModelClient
from .types import (
    ToolConversation as ToolConversation,
)
from .types import (
    ToolDefinition as ToolDefinition,
)
from .types import (
    ToolInvocation as ToolInvocation,
)

__all__ = [
    "CONVERSE_RESERVED",
    "RETRYABLE_SERVER_STATUSES",
    "AttemptFault",
    "AttemptSuccess",
    "AttemptStatus",
    "AttemptTelemetry",
    "InvalidModelResponse",
    "ModelCallFailed",
    "ModelClient",
    "ModelClientError",
    "ModelDeadlineExceeded",
    "ModelIdentityChanged",
    "ToolConversation",
    "ToolDefinition",
    "ToolInvocation",
    "ToolTurn",
    "hash_json",
    "invalid_response",
    "run_attempts",
    "tool_payload",
    "tool_turn_hash",
    "token_counts",
    "usage_from_body",
    "validate_request",
]
