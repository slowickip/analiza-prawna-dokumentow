"""The researcher finds provisions, without judging their fit.

It is the only role holding a corpus search tool, and the only one that may put a
provision forward. What it may not do is rule on
its own shortlist: relevance belongs to the verifier, and the character of a
provision belongs to the analyst.
"""

from __future__ import annotations

from functools import partial

from contract_analyzer.agents.retrieval import read_provision, run_read, run_search
from contract_analyzer.agents.scheduler import (
    FINDER_MAX_PHRASES_PER_TURN,
    TASK_MAX_TURNS,
    Step,
)
from contract_analyzer.agents.session import role_payload
from contract_analyzer.agents.state import UnitContext
from contract_analyzer.agents.tools import (
    PostCandidatesArgs,
    ReadProvisionArgs,
    SearchCorpusArgs,
    parse_tool,
    tools_for,
)
from contract_analyzer.agents.turns import ToolResult, refusal, refused, run_task
from contract_analyzer.model import AttemptTelemetry, ToolInvocation


async def act(context: UnitContext, step: Step) -> None:
    """Run whichever researcher task the scheduler named for this unit."""
    if step.task == "researcher.search":
        await _search(context)
    else:
        raise RuntimeError(f"researcher node reached with {step.task}")


async def _search(context: UnitContext) -> None:
    active = context.active
    payload = {
        **role_payload(active.session, context.call_unit),
        "worksheet": context.worksheet.render("researcher"),
    }
    try:
        outcome = await run_task(
            active,
            role="researcher",
            task="researcher.search",
            unit_id=context.call_unit.unit_id,
            payload=payload,
            tools=tools_for("researcher.search"),
            handle_tool=partial(_handle_search, context),
            max_turns=TASK_MAX_TURNS["researcher.search"],
            max_calls_per_turn=FINDER_MAX_PHRASES_PER_TURN,
        )
    finally:
        # The search is over however it ended, including on an exhausted budget:
        # a unit whose search was cut short must not look like one that has not
        # searched yet, or the scheduler would send it back to search again.
        context.worksheet.search_complete = True
    context.search_turns = outcome.turns
    active.finder_tool_turns += outcome.turns
    if outcome.cap_hit:
        active.finder_budget_exhausted_units += 1


async def _handle_search(
    context: UnitContext,
    invocation: ToolInvocation,
    attempts: tuple[AttemptTelemetry, ...],
) -> ToolResult:
    parsed = parse_tool(
        tools_for("researcher.search"), invocation.name, invocation.arguments
    )
    if isinstance(parsed, dict):
        return refused(parsed, invocation.name, context=context)
    if isinstance(parsed, SearchCorpusArgs):
        return ToolResult(await _search_tool(context, parsed))
    if isinstance(parsed, ReadProvisionArgs):
        return ToolResult(await run_read(context, "researcher", parsed.locator))
    if isinstance(parsed, PostCandidatesArgs):
        return await _post_candidates(context, parsed, invocation.name)
    return refused(refusal("unknown_tool"), invocation.name, context=context)


async def _search_tool(
    context: UnitContext, args: SearchCorpusArgs
) -> dict[str, object]:
    result, candidates = await run_search(
        context.active, args.phrase, unit_id=context.call_unit.unit_id
    )
    # Any error payload means the phrase was refused before the corpus was
    # queried -- an empty sanitised query still carries an empty "results" list,
    # so keying off that key alone recorded a search nobody ran, and the
    # no-basis guard below would then accept a conclusion drawn from nothing.
    if "error" in result or "results" not in result:
        return result
    context.active.finder_search_calls += 1
    context.worksheet.add_search(
        phrase=args.phrase,
        result_locators=[candidate.locator for candidate in candidates],
    )
    context.locator_ledger.update(candidate.locator for candidate in candidates)
    return result


async def _post_candidates(
    context: UnitContext, args: PostCandidatesArgs, tool_name: str
) -> ToolResult:
    # Posting nothing is a finding about the corpus, so it has to rest on a look
    # at the corpus. Without this the floor of zero would also authorise giving up
    # on the first turn, which would cost completeness to buy time.
    if not args.candidates and not any(
        entry.kind == "search" for entry in context.worksheet.entries
    ):
        return refused(
            refusal("no_search_before_empty_commit"), tool_name, context=context
        )
    locators = [candidate.locator for candidate in args.candidates]
    if len(set(locators)) != len(locators):
        return refused(refusal("duplicate_candidate"), tool_name, context=context)
    ledger = context.locator_ledger
    claimed = {
        locator
        for candidate in args.candidates
        for locator in (candidate.locator, *candidate.supporting_locators)
    }
    if not claimed <= ledger:
        return refused(
            refusal("locator_not_in_provenance_ledger"), tool_name, context=context
        )
    records = []
    for locator in locators:
        record = await read_provision(context.active, locator)
        if record is None:
            return refused(refusal("unknown_locator"), tool_name, context=context)
        records.append(record)
    entries = [
        context.worksheet.add_candidate(
            locator=record.locator,
            act_identifier=record.act_identifier,
            snapshot_id=context.active.services.corpus.snapshot.id,
            act_force=record.act_force,
            provision_force=record.provision_force,
            why=candidate.why,
            supporting_locators=candidate.supporting_locators,
        )
        for candidate, record in zip(args.candidates, records, strict=True)
    ]
    context.candidate_count += len(entries)
    return ToolResult(
        {"candidate_ids": [entry.id for entry in entries]}, committed=True
    )
