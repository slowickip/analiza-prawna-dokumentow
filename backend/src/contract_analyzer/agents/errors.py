"""Pipeline failures that carry the machine error code the API reports."""

from __future__ import annotations


class RunPipelineError(Exception):
    """A run failure identified by a stable lower-snake-case code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class BudgetExhausted(RunPipelineError):
    """The run's token or wall budget ran out before the work finished."""


class InvalidChatCitation(RunPipelineError):
    """The explanation cited a finding outside the requested selection."""


class MessageNotRoutable(RunPipelineError):
    """The router committed no reading of the reader's message.

    Routing is one model call with one commit tool and no corpus access, so a
    message that ends without a commit is a model failure, not a quiet question:
    treating it as one would answer something the reader did not ask.
    """


class ChatOutOfScope(RunPipelineError):
    """The explanation was asked for something outside the run's findings."""


class RetrievalDependencyUnavailable(RunPipelineError):
    """The embedding service failed while a run was retrieving candidates.

    Carries dependency_unavailable, the code the readiness probe already reports
    this dependency under, rather than letting EmbeddingError escape as
    internal_error -- which the contract reserves for a defect in this server. A
    run that dies because a self-hosted service went away is not an artefact
    defect, and error analysis that cannot tell the two apart attributes an
    outage to the thing being measured.
    """


def first_leaf(error: BaseExceptionGroup[BaseException]) -> BaseException:
    """The first leaf of a fan-out failure, keeping its type and instance."""
    leaf: BaseException = error
    while isinstance(leaf, BaseExceptionGroup):
        leaf = leaf.exceptions[0]
    return leaf
