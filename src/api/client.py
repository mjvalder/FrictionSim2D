"""HTTP client for the FrictionSim2D REST API.

Provides :class:`FrictionHTTPClient`, a drop-in replacement for
:class:`~src.data.database.FrictionDB` that talks to the REST API instead
of connecting directly to PostgreSQL.

Usage::

    from src.api.client import FrictionHTTPClient

    client = FrictionHTTPClient(
        api_url="https://db.example.com",
        api_key="sk_abc123...",
    )

    # Same interface as FrictionDB
    df = client.query(material="h-MoS2")
    client.upload_record(record)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FrictionHTTPClient:
    """HTTP client for the FrictionSim2D REST API.

    Mirrors the public API of :class:`~src.data.database.FrictionDB` so
    callers can swap implementations transparently.

    Args:
        api_url: Base URL of the API (e.g. ``"http://localhost:8000"``).
        api_key: Optional API key for authenticated endpoints.
        timeout: Request timeout in seconds.

    Raises:
        ImportError: If ``httpx`` is not installed.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        try:
            import httpx  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the HTTP client. "
                "Install with: pip install httpx"
            ) from exc

        self._base = api_url.rstrip("/")
        self._api_key = api_key.strip() or None if api_key is not None else None
        self._client = httpx.Client(timeout=timeout)

    # -- Internal helpers -----------------------------------------------------

    def _headers(self, auth: bool = False) -> Dict[str, str]:
        """Build request headers, optionally including the API key."""
        headers: Dict[str, str] = {"Accept": "application/json"}
        if auth and self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _raise_for_status(self, resp) -> None:
        """Raise a descriptive error on non-2xx responses."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except (AttributeError, TypeError, ValueError):
                detail = resp.text
            raise RuntimeError(
                f"API error {resp.status_code}: {detail}"
            )

    # -- Upload ----------------------------------------------------------------

    def upload_result(self, **kwargs) -> Optional[int]:
        """Stage a new result via the API.

        Accepts the same keyword arguments as
        :meth:`~src.data.database.FrictionDB.upload_result`.

        Returns:
            Database row ID of the staged result, or ``None`` if the API
            reports ``409 Conflict`` -- a duplicate ``time_series_hash``
            already exists in the central DB (the same result uploaded
            twice). Mirrors :meth:`~src.data.database.FrictionDB.upload_result`'s
            ``Optional[int]`` return for the same outcome, so callers can
            tell "already there" apart from a genuine server error with
            an ``is None`` check instead of parsing the exception message.
            Any other 4xx/5xx still raises :class:`RuntimeError` via
            :meth:`_raise_for_status`.
        """
        # Remove keys that the API doesn't accept via body
        kwargs.pop("uploader", None)
        kwargs.pop("status", None)
        # Filter None values
        body = {k: v for k, v in kwargs.items() if v is not None}

        resp = self._client.post(
            self._url("/results"),
            json=body,
            headers=self._headers(auth=True),
        )
        if resp.status_code == 409:
            return None
        self._raise_for_status(resp)
        return resp.json()["id"]

    def upload_record(self, record: Any, uploader: Optional[str] = None) -> Optional[int]:
        """Upload a :class:`~src.data.models.ResultRecord` via the API.

        Returns ``None`` on a duplicate upload -- see :meth:`upload_result`.
        """
        _ = uploader
        d = record.to_upload_dict()
        d.pop("uploader", None)
        d.pop("status", None)
        return self.upload_result(**d)

    # -- Query ----------------------------------------------------------------

    def query(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        material: Optional[str] = None,
        family: Optional[str] = None,
        model: Optional[str] = None,
        set_label: Optional[str] = None,
        potential_type: Optional[str] = None,
        layers: Optional[int] = None,
        uploader: Optional[str] = None,
        force_range: Optional[tuple] = None,
        pressure_range: Optional[tuple] = None,
        temperature_range: Optional[tuple] = None,
        angle_range: Optional[tuple] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: str = "uploaded_at DESC",
    ) -> "pandas.DataFrame":
        """Query results via the API. Returns a pandas DataFrame."""
        import pandas as pd  # pylint: disable=import-outside-toplevel

        params: Dict[str, Any] = {"order_by": order_by}
        for key, value in (("material", material), ("family", family),
                           ("model", model), ("set_label", set_label),
                           ("potential_type", potential_type),
                           ("layers", layers), ("uploader", uploader),
                           ("status", status), ("limit", limit)):
            if value is not None:
                params[key] = value
        for prefix, span in (("force", force_range), ("pressure", pressure_range),
                             ("temp", temperature_range), ("angle", angle_range)):
            if span is not None:
                params[f"{prefix}_min"], params[f"{prefix}_max"] = span

        resp = self._client.get(self._url("/results"), params=params,
                                headers=self._headers())
        self._raise_for_status(resp)
        return pd.DataFrame(resp.json()["results"])

    # -- Single result -------------------------------------------------------

    def get_result(self, result_id: int) -> Dict[str, Any]:
        """Retrieve a single result by ID."""
        resp = self._client.get(
            self._url(f"/results/{result_id}"),
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # -- Statistics -----------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        resp = self._client.get(
            self._url("/statistics"),
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # -- Materials & conditions -----------------------------------------------

    def list_materials(self) -> List[str]:
        """List distinct material names."""
        resp = self._client.get(
            self._url("/materials"),
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()["materials"]

    def get_conditions(self) -> Dict[str, Any]:
        """Return parameter ranges."""
        resp = self._client.get(
            self._url("/conditions"),
            headers=self._headers(),
        )
        self._raise_for_status(resp)
        return resp.json()

    # -- Staging pipeline (curator actions) -----------------------------------

    def validate_staged(self, row_id: int) -> Dict[str, Any]:
        """Run automated validation on a staged result."""
        resp = self._client.post(
            self._url(f"/results/{row_id}/validate"),
            headers=self._headers(auth=True),
        )
        self._raise_for_status(resp)
        return resp.json()

    def publish(self, row_id: int) -> bool:
        """Promote a validated result to published."""
        resp = self._client.post(
            self._url(f"/results/{row_id}/publish"),
            headers=self._headers(auth=True),
        )
        self._raise_for_status(resp)
        return True

    def reject(self, row_id: int, reason: Optional[str] = None) -> bool:
        """Reject a result."""
        resp = self._client.post(
            self._url(f"/results/{row_id}/reject"),
            json={"reason": reason},
            headers=self._headers(auth=True),
        )
        self._raise_for_status(resp)
        return True

    # -- Cleanup --------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
