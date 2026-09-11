"""Provider client protocols."""

from typing import Protocol

from .attempts import ToolTurn
from .types import ToolConversation


class ModelClient(Protocol):
    """The provider surface the pipeline depends on."""

    @property
    def max_attempts(self) -> int: ...

    async def converse(self, conversation: ToolConversation) -> ToolTurn: ...

    async def aclose(self) -> None: ...
