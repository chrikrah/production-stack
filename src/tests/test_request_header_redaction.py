"""The session-extraction debug line must not defeat TokenRedactionFilter.

TokenRedactionFilter only inspects ``record.args`` for a Starlette ``Headers``
object, so a call site that formats the headers into the message string leaves
nothing for it to redact.
"""

import logging

from starlette.datastructures import Headers

from vllm_router import log


def test_formatted_headers_are_not_redacted_but_a_lazy_arg_is():
    redaction = log.TokenRedactionFilter()
    headers = Headers(
        {"authorization": "Bearer super-secret-token", "host": "example.invalid"}
    )

    formatted = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg=f"Debug session extraction - Request headers: {dict(headers)}",
        args=None,
        exc_info=None,
    )
    redaction.filter(formatted)
    assert "super-secret-token" in formatted.getMessage(), (
        "a pre-formatted dict carries nothing in record.args, so the filter "
        "has nothing to act on; this is what the call site must avoid"
    )

    lazy = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg="Debug session extraction - Request headers: %s",
        args=(headers,),
        exc_info=None,
    )
    redaction.filter(lazy)
    assert "super-secret-token" not in lazy.getMessage()
    assert "Bearer ****" in lazy.getMessage()


def test_session_extraction_logs_headers_as_a_lazy_arg():
    """The call site itself, read from source, must pass the Headers object."""
    import inspect

    from vllm_router.services.request_service import request as request_module

    source = inspect.getsource(request_module.route_general_request)
    assert 'logger.debug("Debug session extraction - Request headers: %s"' in source, (
        "the headers must reach the logger as a lazy argument, not formatted "
        "into the message, or TokenRedactionFilter cannot redact them"
    )
    assert "Request headers: {dict(request.headers)}" not in source
