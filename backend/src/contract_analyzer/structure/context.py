"""Reference-derived context selection."""

from contract_analyzer.domain import ReferenceRecord, ReferenceStatus, Unit

from .segmentation import _is_fallback_units


def context_for(
    unit_id: str,
    units: list[Unit],
    references: list[ReferenceRecord],
) -> list[Unit]:
    """Units this unit resolvably refers to, in reference order, deduplicated.

    Empty for a window-fallback document, which has no markers to cite, and empty
    for a single-unit document, which has nothing to cite. Both are ordinary
    results: the ON arm runs on such documents and records that it received no
    context, which the evaluation protocol reads as its noise floor rather than as
    a failure. This returns absence without distinguishing its cause; the run
    record's context_edge_count is what an analysis partitions on.
    """
    if _is_fallback_units(units):
        return []

    units_by_id = {unit.id: unit for unit in units}
    targets: list[Unit] = []
    seen: set[str] = set()
    for reference in references:
        if reference.citing_unit_id != unit_id:
            continue
        if reference.status is not ReferenceStatus.RESOLVED:
            continue
        if reference.target_unit_id is None:
            continue
        if reference.target_unit_id == unit_id:
            continue
        if reference.target_unit_id in seen:
            continue
        target = units_by_id.get(reference.target_unit_id)
        if target is None:
            continue
        seen.add(reference.target_unit_id)
        targets.append(target)
    return targets
