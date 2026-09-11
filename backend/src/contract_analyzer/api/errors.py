"""HTTP error catalogue and handlers."""

from __future__ import annotations

import logging
from typing import Any, NoReturn

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from contract_analyzer.api.schemas import Error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

# (HTTP status | None, Polish message) for every error code. Status is None for codes
# that only appear inside finished runs or telemetry, never directly returned as
# HTTP responses.
_ERRORS: dict[str, tuple[int | None, str]] = {
    # ingestion
    "unsupported_input": (415, "Nieobsługiwany format pliku."),
    "empty_input": (422, "Plik jest pusty lub nie zawiera tekstu."),
    "corrupt_input": (422, "Plik jest uszkodzony lub nie może zostać odczytany."),
    "limit_breach": (413, "Plik przekracza dopuszczalny limit rozmiaru."),
    "missing_tool": (503, "Wymagane narzędzie nie jest dostępne na serwerze."),
    "conversion_failed": (422, "Konwersja dokumentu nie powiodła się."),
    "ocr_failed": (422, "Rozpoznawanie tekstu (OCR) nie powiodło się."),
    # structure
    "empty_document": (422, "Dokument nie zawiera treści do analizy."),
    "invalid_window": (422, "Nieprawidłowa konfiguracja okna segmentacji."),
    "no_units": (422, "Nie udało się wyodrębnić jednostek z dokumentu."),
    # model client
    "model_transport_error": (None, "Nie udało się połączyć z dostawcą modelu."),
    "model_response_invalid_envelope": (
        None,
        "Odpowiedź modelu miała nieprawidłową strukturę.",
    ),
    "model_response_schema_invalid": (
        None,
        "Odpowiedź modelu nie spełniła wymaganego schematu.",
    ),
    "model_http_error": (None, "Dostawca modelu zwrócił błąd HTTP."),
    "model_identity_changed": (
        None,
        "Dostawca modelu zmienił identyfikator modelu w trakcie analizy.",
    ),
    "model_response_missing_usage": (
        None,
        "Odpowiedź modelu nie zawierała informacji o zużyciu tokenów.",
    ),
    # pipeline
    "retrieval_query_empty": (None, "Zapytanie do korpusu było puste."),
    "unit_graph_recursion_exceeded": (
        None,
        "Analiza jednostki przekroczyła limit przebiegu.",
    ),
    "budget_exhausted": (
        None,
        "Analiza przerwana: wyczerpano budżet uruchomienia.",
    ),
    "chat_citation_unknown": (
        422,
        "Pytanie odwołuje się do ustalenia spoza tej analizy.",
    ),
    "evidence_locator_unknown": (
        None,
        "Wskazany przepis nie występuje w zamrożonej migawce korpusu.",
    ),
    # storage
    "process_restarted": (None, "Serwer został zrestartowany w trakcie analizy."),
    # server layer
    "content_expired": (410, "Treść dokumentu została usunięta lub wygasła."),
    "not_found": (404, "Nie znaleziono wskazanego zasobu."),
    "run_already_active": (409, "Inna analiza jest w trakcie wykonywania."),
    "run_not_finished": (409, "Analiza nadrzędna nie została jeszcze zakończona."),
    "evaluation_batch_open": (409, "Trwa końcowa seria pomiarowa."),
    "run_already_terminal": (409, "Analiza została już zakończona."),
    "out_of_scope": (422, "Pytanie wykracza poza zakres wyjaśniania ustaleń."),
    "message_not_routable": (
        422,
        "System nie rozpoznał, czego dotyczy wiadomość. Sformułuj ją inaczej.",
    ),
    "invalid_request": (422, "Treść żądania jest nieprawidłowa."),
    "dependency_unavailable": (
        503,
        "Wymagana zależność zewnętrzna nie jest dostępna.",
    ),
    "internal_error": (500, "Wystąpił błąd wewnętrzny serwera."),
}

_HTTP_STATUS: dict[str, int] = {
    code: status for code, (status, _) in _ERRORS.items() if status is not None
}
_MESSAGES_PL: dict[str, str] = {code: message for code, (_, message) in _ERRORS.items()}


class _ApiError(Exception):
    """Raised inside endpoints to produce a structured error response."""

    def __init__(self, code: str, *, detail: dict[str, Any] | None = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def _raise_api(code: str, *, detail: dict[str, Any] | None = None) -> NoReturn:
    raise _ApiError(code, detail=detail)


def _normalise_code(code: str) -> str:
    """Collapse the dynamic model_http_<status> family onto its declared name.

    model.py mints a code per HTTP status, so no fixed table can list them.
    openapi.yaml already declares model_http_error as the finite representation
    of that family; this is where the collapse happens.
    """
    if code.startswith("model_http_"):
        return "model_http_error"
    return code


def _message_for(code: str) -> str:
    """Polish message for a stored or raised code, never raising on an unknown one.

    A run that failed is still readable: an unrecognised code must not turn
    GET /runs/{id} into a 500 for the rest of that run's life.
    """
    normalised = _normalise_code(code)
    if normalised in _MESSAGES_PL:
        return _MESSAGES_PL[normalised]
    logger.warning("no Polish message for error code %r", code)
    return _MESSAGES_PL["internal_error"]


def _error_response(code: str, *, detail: dict[str, Any] | None = None) -> JSONResponse:
    status = _HTTP_STATUS[_normalise_code(code)]
    message_pl = _message_for(code)
    return JSONResponse(
        status_code=status,
        content=Error(
            code=code,
            message_pl=message_pl,
            detail=detail,
        ).model_dump(exclude_none=True),
    )


# ---------------------------------------------------------------------------


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(_ApiError)
    async def api_error_handler(request: Request, exc: _ApiError) -> JSONResponse:
        return _error_response(exc.code, detail=exc.detail)

    # FastAPI's default body for a malformed request is a different shape from
    # Error, which would put two bodies behind one status. Its "input" field also
    # echoes the submitted value back, and on /message that value is the user's
    # question and history. Neither is reflected here: the code alone says what
    # was wrong, and the validation detail goes to the log.
    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info(
            "Request validation failed for %s %s", request.method, request.url.path
        )
        # Which field failed and why, never the value that failed. "input" in
        # FastAPI's default body would reflect the submitted content back, and on
        # /message that content is the user's message and history.
        fields = [
            {
                "loc": [str(part) for part in issue.get("loc", ())],
                "type": issue.get("type", ""),
            }
            for issue in exc.errors()
        ]
        return _error_response("invalid_request", detail={"fields": fields})

    # Global handler to prevent untyped exceptions from escaping as 500
    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception in %s %s", request.method, request.url.path
        )
        return _error_response("internal_error")
