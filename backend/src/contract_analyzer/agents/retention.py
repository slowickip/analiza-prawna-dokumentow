"""What a run keeps of its own prose, and for how long.

The synthesis and the finished worksheets are derived from the document, so they
are held in the same time-limited store as the document text and a purge takes
them along. Nothing here reaches Postgres: the store is in process, keyed by an
identifier, and it overwrites the bytes when it drops them.
"""

from __future__ import annotations

from uuid import UUID

from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.state import StoredText
from contract_analyzer.storage import ContentExpired


class TextRetention:
    """The run's derived prose, addressable while the retention window is open."""

    def __init__(self, services: AnalysisServices) -> None:
        self._services = services
        # Both outlive the active run on purpose: a grouping and a worksheet are
        # read after the run is retired, until the window closes.
        self._synthesis: dict[UUID, StoredText] = {}
        self._worksheets: dict[UUID, dict[str, StoredText]] = {}

    def keep_synthesis(
        self, *, run_id: UUID, document_id: UUID, payload: bytes
    ) -> None:
        if self._services.text_store is None:
            return
        self._synthesis[run_id] = StoredText(
            document_id=document_id,
            text_key=self._services.text_store.put(payload),
        )

    def synthesis_bytes(self, run_id: UUID) -> bytes | None:
        """The stored grouping, or None once it is gone.

        None covers three cases the caller treats alike: the run predates
        synthesis, the installation runs without a text store, and the retention
        window closed.
        """
        stored = self._synthesis.get(run_id)
        if stored is None or self._services.text_store is None:
            return None
        try:
            return self._services.text_store.get(stored.text_key)
        except ContentExpired:
            self._synthesis.pop(run_id, None)
            return None

    def keep_worksheet(
        self, *, run_id: UUID, document_id: UUID, unit_id: str, payload: bytes
    ) -> None:
        if self._services.text_store is None:
            return
        self._worksheets.setdefault(run_id, {})[unit_id] = StoredText(
            document_id=document_id,
            text_key=self._services.text_store.put(payload),
        )

    def worksheet_bytes(self, run_id: UUID, unit_id: str) -> bytes | None:
        """The finished worksheet of one call unit, while it is still retained."""
        stored = self._worksheets.get(run_id, {}).get(unit_id)
        if stored is None or self._services.text_store is None:
            return None
        try:
            return self._services.text_store.get(stored.text_key)
        except ContentExpired:
            self._worksheets.get(run_id, {}).pop(unit_id, None)
            return None

    def worksheet_unit_ids(self, run_id: UUID) -> tuple[str, ...]:
        return tuple(self._worksheets.get(run_id, {}))

    def drop_document(self, document_id: UUID) -> None:
        """Forget everything derived from one document."""
        for run_id, stored in list(self._synthesis.items()):
            if stored.document_id == document_id:
                self._delete(stored)
                self._synthesis.pop(run_id, None)
        for run_id, units in list(self._worksheets.items()):
            for unit_id, stored in list(units.items()):
                if stored.document_id == document_id:
                    self._delete(stored)
                    units.pop(unit_id, None)
            if not units:
                self._worksheets.pop(run_id, None)

    def drop_all(self) -> None:
        for stored in self._synthesis.values():
            self._delete(stored)
        self._synthesis.clear()
        for units in self._worksheets.values():
            for stored in units.values():
                self._delete(stored)
        self._worksheets.clear()

    def _delete(self, stored: StoredText) -> None:
        if self._services.text_store is not None:
            self._services.text_store.delete(stored.text_key)
