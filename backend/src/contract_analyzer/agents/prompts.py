"""The prompt bundle: what each task says, and the hash parity compares.

``prompt_bundle_version`` is a parity dimension, so the hash has to cover every
prompt the package ships. ``PROMPT_FILES`` is the single list the loader, the
hash and the packaging test all read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib import resources

from contract_analyzer.agents.schema import FrozenModel

PROMPT_FILES: tuple[str, ...] = (
    "researcher_search.md",
    "analyst_characterise.md",
    "verifier_relevance.md",
    "verifier_relation.md",
    "synthesise.md",
    "explain.md",
    "researcher_question.md",
    "route_message.md",
)

# Which prompt drives which task. The task name is also the recorded
# prompt_version, so an attempt record names the task that spent the tokens.
TASK_PROMPTS: Mapping[str, str] = {
    "researcher.search": "researcher_search.md",
    "analyst.characterise": "analyst_characterise.md",
    "verifier.relevance": "verifier_relevance.md",
    "verifier.relation": "verifier_relation.md",
    "synthesizer.group": "synthesise.md",
}


class PromptBundle(FrozenModel):
    version: str
    prompts: dict[str, str]


def load_prompt_bundle() -> PromptBundle:
    """Read every shipped prompt and stamp the bundle with its hash."""
    prompts: dict[str, str] = {}
    package = "contract_analyzer.prompts"
    for filename in PROMPT_FILES:
        prompts[filename] = (
            resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        )
    return PromptBundle(version=hash_prompt_bundle(prompts), prompts=prompts)


def hash_prompt_bundle(prompts: Mapping[str, str] | None = None) -> str:
    """Digest the bundle, reading it from the package when none is supplied."""
    if prompts is None:
        return load_prompt_bundle().version
    material = "".join(prompts[name] for name in PROMPT_FILES)
    return hashlib.sha256(material.encode()).hexdigest()
