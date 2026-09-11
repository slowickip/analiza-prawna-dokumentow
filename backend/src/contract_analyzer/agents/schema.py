"""The strict base model every agent-facing schema is built on."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from contract_analyzer.model import ToolDefinition


class FrozenModel(BaseModel):
    """Immutable and closed: an unexpected key is a rejected answer, not extra.

    Tool arguments and model answers are validated against these models, so a
    field the schema does not name has to fail rather than be dropped silently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class ToolSpec:
    """One model-visible tool contract and its safe argument parser."""

    name: str
    description: str
    arguments: type[BaseModel]
    commit: bool = False

    @cached_property
    def _schema(self) -> dict[str, Any]:
        """The argument schema, reflected once.

        The specs are module constants, and pydantic walked the argument model
        to build this 120 to 180 times a run for a schema that cannot change.
        """
        return self.arguments.model_json_schema()

    def definition(self) -> ToolDefinition:
        """The tool contract for one turn, owned by its caller.

        The schema is cached but copied out: ToolDefinition is frozen only at
        the top level, so handing the same nested dictionaries to every turn
        would let one caller edit what every later run is offered, while the
        recorded bundle digest, which is rebuilt from the models, kept saying
        otherwise.
        """
        return ToolDefinition(self.name, self.description, deepcopy(self._schema))

    def parse(self, arguments: str) -> BaseModel | dict[str, object]:
        """Parse one call or return a refusal without echoing its input."""
        try:
            return self.arguments.model_validate_json(arguments)
        except ValidationError as error:
            return validation_refusal(error)


def validation_refusal(error: ValidationError) -> dict[str, object]:
    """The refusal for arguments the schema rejects, without echoing input.

    Shared by the direct parser above and the graph tool wrappers, so a model
    corrected by either path sees the same code and field detail.
    """
    return {
        "error": "invalid_arguments",
        "detail": [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "type": item["type"],
                "msg": item["msg"],
            }
            for item in error.errors(include_input=False)
        ],
    }
