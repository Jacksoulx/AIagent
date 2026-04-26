import time
from typing import Any, List, Optional, Sequence

import requests


class RpcClientError(RuntimeError):
    """Base error for JSON-RPC client failures."""


class MissingRpcUrlError(RpcClientError):
    """Raised when a required RPC endpoint is missing."""


class RpcTimeoutError(RpcClientError):
    """Raised when an RPC request times out."""


class RpcHttpError(RpcClientError):
    """Raised when the RPC endpoint returns a non-200 response."""


class RpcResponseError(RpcClientError):
    """Raised when the RPC endpoint returns an application-level error."""


class RpcMalformedResponseError(RpcClientError):
    """Raised when the RPC response cannot be interpreted safely."""


class JsonRpcClient:
    """Small JSON-RPC client with explicit error handling."""

    def __init__(
        self,
        rpc_url: str,
        timeout_sec: float = 10.0,
        max_retries: int = 2,
        retry_backoff_sec: float = 0.5,
        retry_http_statuses: Optional[Sequence[int]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not rpc_url or not rpc_url.strip():
            raise MissingRpcUrlError(
                "Environment variable ETH_RPC_URL not found. "
                "Please set it to an Ethereum JSON-RPC endpoint."
            )

        self.rpc_url = rpc_url.strip()
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.retry_http_statuses = tuple(retry_http_statuses or (429, 500, 502, 503, 504))
        self.session = session or requests.Session()
        self._request_id = 0

    def call(self, method: str, params: Optional[List[Any]] = None) -> Any:
        """Execute a JSON-RPC method call and return its result."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or [],
        }

        last_error: Optional[Exception] = None
        total_attempts = self.max_retries + 1

        for attempt_index in range(total_attempts):
            attempt_number = attempt_index + 1
            try:
                response = self.session.post(
                    self.rpc_url,
                    json=payload,
                    timeout=self.timeout_sec,
                )
            except requests.Timeout as exc:
                last_error = exc
                if attempt_number < total_attempts:
                    time.sleep(self.retry_backoff_sec * attempt_number)
                    continue
                raise RpcTimeoutError(
                    f"RPC request timed out after {self.timeout_sec:.1f}s for method {method} "
                    f"after {attempt_number} attempt(s)."
                ) from exc
            except requests.RequestException as exc:
                last_error = exc
                if attempt_number < total_attempts:
                    time.sleep(self.retry_backoff_sec * attempt_number)
                    continue
                raise RpcClientError(
                    f"RPC request failed for method {method} after {attempt_number} attempt(s): {exc}"
                ) from exc

            if response.status_code != 200:
                if (
                    response.status_code in self.retry_http_statuses
                    and attempt_number < total_attempts
                ):
                    time.sleep(self.retry_backoff_sec * attempt_number)
                    continue
                raise RpcHttpError(
                    f"RPC endpoint returned HTTP {response.status_code} for method {method} "
                    f"on attempt {attempt_number}/{total_attempts}."
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise RpcMalformedResponseError(
                    f"RPC endpoint returned non-JSON content for method {method}."
                ) from exc

            if not isinstance(payload, dict):
                raise RpcMalformedResponseError(
                    f"RPC response for method {method} was not a JSON object."
                )

            error = payload.get("error")
            if error:
                if isinstance(error, dict):
                    code = error.get("code", "unknown")
                    message = error.get("message", "unknown error")
                    raise RpcResponseError(
                        f"JSON-RPC error for method {method}: code={code}, message={message}"
                    )
                raise RpcResponseError(f"JSON-RPC error for method {method}: {error}")

            if "result" not in payload:
                raise RpcMalformedResponseError(
                    f"RPC response for method {method} does not contain a result field."
                )

            return payload["result"]

        raise RpcClientError(f"RPC request failed for method {method}: {last_error}")
