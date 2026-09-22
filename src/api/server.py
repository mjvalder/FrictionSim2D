"""FastAPI REST server for the FrictionSim2D shared database.

Endpoints
---------
Public (no auth):
    GET  /results          — Query results with filters
    GET  /results/{id}     — Single result detail
    GET  /statistics       — Aggregate statistics
    GET  /materials        — List distinct materials
    GET  /conditions       — Parameter ranges in the dataset

Authenticated (X-API-Key header):
    POST /results          — Stage a new result
    POST /results/{id}/publish  — Promote validated → published (curator)
    POST /results/{id}/reject   — Reject a result (curator)
    POST /results/{id}/validate — Run automated validation

Health:
    GET  /health           — Liveness check
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import get_db, optional_api_key, require_api_key, set_db

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(db=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db: An existing :class:`~src.data.database.FrictionDB`. If ``None``, the
            central database from ``settings.database`` is used.
    """
    if db is None:
        # Deferred: `..data.database` pulls in `..data.models`, which imports
        # numpy. A caller that supplies its own `db` (schema generation, a
        # stub in tests) should not pay that import cost just for creating
        # the app.
        from ..data.database import central_db  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        db = central_db()

    set_db(db)
    return app


app = FastAPI(
    title="FrictionSim2D API",
    description="REST interface for the shared friction simulation database.",
    version="0.1.0",
)

# CORS — permissive for academic/internal use; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResultCreate(BaseModel):
    """Schema for staging a new result via the API (schema 3)."""
    material: str
    materials: List[str] = []
    model: Optional[str] = None
    family: Optional[str] = None
    contact: Optional[str] = None
    run_uuid: Optional[str] = None
    result_uuid: Optional[str] = None
    set_label: Optional[str] = None
    source_host: Optional[str] = None
    stack_type: Optional[str] = None
    layers: Optional[int] = None
    size_x: Optional[float] = None
    size_y: Optional[float] = None
    temperature: Optional[float] = None
    tip_material: Optional[str] = None
    tip_radius: Optional[float] = None
    substrate_material: Optional[str] = None
    substrate_amorphous: Optional[bool] = None
    potential_type: Optional[str] = None
    potential_name: Optional[str] = None
    pot_sha256: Optional[str] = None
    git_commit: Optional[str] = None
    force_nN: Optional[float] = None
    pressure_gpa: Optional[float] = None
    scan_angle: Optional[float] = None
    scan_speed: Optional[float] = None
    mean_cof: Optional[float] = None
    std_cof: Optional[float] = None
    mean_lf: Optional[float] = None
    std_lf: Optional[float] = None
    mean_nf: Optional[float] = None
    std_nf: Optional[float] = None
    mean_lfx: Optional[float] = None
    std_lfx: Optional[float] = None
    mean_lfy: Optional[float] = None
    std_lfy: Optional[float] = None
    ntimesteps: Optional[int] = None
    time_series_hash: Optional[str] = None
    is_complete: bool = True
    extras: Dict[str, Any] = {}
    notes: Optional[str] = None
    data_url: Optional[str] = None


class ResultResponse(BaseModel):
    """One central row. Field-for-field `ResultRecord`, with `uploaded_at`
    serialised as an ISO string."""
    id: Optional[int] = None
    uploaded_at: Optional[str] = None
    uploader: Optional[str] = None
    contact: Optional[str] = None
    run_uuid: Optional[str] = None
    result_uuid: Optional[str] = None
    set_label: Optional[str] = None
    source_host: Optional[str] = None
    model: Optional[str] = None
    family: Optional[str] = None
    materials: List[str] = []
    material: Optional[str] = None
    stack_type: Optional[str] = None
    layers: Optional[int] = None
    size_x: Optional[float] = None
    size_y: Optional[float] = None
    temperature: Optional[float] = None
    tip_material: Optional[str] = None
    tip_radius: Optional[float] = None
    substrate_material: Optional[str] = None
    substrate_amorphous: Optional[bool] = None
    potential_type: Optional[str] = None
    potential_name: Optional[str] = None
    pot_sha256: Optional[str] = None
    git_commit: Optional[str] = None
    force_nN: Optional[float] = None
    pressure_gpa: Optional[float] = None
    scan_angle: Optional[float] = None
    scan_speed: Optional[float] = None
    mean_cof: Optional[float] = None
    std_cof: Optional[float] = None
    mean_lf: Optional[float] = None
    std_lf: Optional[float] = None
    mean_nf: Optional[float] = None
    std_nf: Optional[float] = None
    mean_lfx: Optional[float] = None
    std_lfx: Optional[float] = None
    mean_lfy: Optional[float] = None
    std_lfy: Optional[float] = None
    ntimesteps: Optional[int] = None
    time_series_hash: Optional[str] = None
    is_complete: Optional[bool] = None
    extras: Dict[str, Any] = {}
    status: Optional[str] = None
    notes: Optional[str] = None
    data_url: Optional[str] = None


class QueryResponse(BaseModel):
    """Paginated query response."""
    count: int
    results: List[ResultResponse]


class StatisticsResponse(BaseModel):
    """Aggregate statistics."""
    total_rows: int
    by_material: Dict[str, int]
    by_type: Dict[str, int]
    cof_global_mean: Optional[float]


class MaterialsResponse(BaseModel):
    """List of distinct materials."""
    materials: List[str]


class ConditionsResponse(BaseModel):
    """Parameter ranges and category counts in the dataset."""
    force_nN: Optional[Dict[str, float]] = None
    temperature: Optional[Dict[str, float]] = None
    scan_angle: Optional[Dict[str, float]] = None
    pressure_gpa: Optional[Dict[str, float]] = None
    layers: Optional[Dict[str, int]] = None
    family: Optional[Dict[str, int]] = None
    model: Optional[Dict[str, int]] = None


FloatRange = Dict[str, float]
IntRange = Dict[str, int]


class RejectRequest(BaseModel):
    """Body for rejecting a result."""
    reason: Optional[str] = None


class ValidationResponse(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class UploadResponse(BaseModel):
    """Response after staging a result."""
    id: Optional[int] = None
    status: str


# ---------------------------------------------------------------------------
# Helper to convert DataFrame row to ResultResponse
# ---------------------------------------------------------------------------


def _row_to_response(row: dict) -> ResultResponse:
    """Build the response straight from the row: one field list, not two."""
    data = {field: row.get(field) for field in ResultResponse.model_fields.keys()}
    uploaded = data.get('uploaded_at')
    data['uploaded_at'] = uploaded.isoformat() if hasattr(uploaded, 'isoformat') \
        else (str(uploaded) if uploaded is not None else None)
    data['materials'] = list(row.get('materials') or [])
    data['extras'] = dict(row.get('extras') or {})
    return ResultResponse(**data)


def _filter_visible_rows(df, viewer_name: Optional[str],
                        requested_status: Optional[str] = None):
    """Return only rows visible to the current viewer.

    Public requests can only see published results. Authenticated requests
    may see all rows by default or apply an explicit status filter.
    """
    if df.empty:
        return df

    status_filter = requested_status
    if viewer_name is None:
        if requested_status is None:
            status_filter = 'published'
        elif requested_status != 'published':
            return df.iloc[0:0]

    if status_filter is None or 'status' not in df.columns:
        return df
    return df[df['status'] == status_filter]


def _statistics_from_rows(df) -> StatisticsResponse:
    """Build aggregate statistics from a filtered result DataFrame."""
    if df.empty:
        return StatisticsResponse(
            total_rows=0,
            by_material={},
            by_type={},
            cof_global_mean=None,
        )

    materials = df['material'].dropna().value_counts().to_dict()
    by_family = df['family'].dropna().value_counts().to_dict() \
        if 'family' in df.columns else {}
    cof_values = df['mean_cof'].dropna()
    cof_global_mean = float(cof_values.mean()) if not cof_values.empty else None
    return StatisticsResponse(
        total_rows=len(df),
        by_material=materials,
        by_type=by_family,
        cof_global_mean=cof_global_mean,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Public endpoints (read-only, no auth required)
# ---------------------------------------------------------------------------


@app.get("/results", response_model=QueryResponse)
def query_results(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    material: Optional[str] = Query(None),
    family: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    set_label: Optional[str] = Query(None),
    potential_type: Optional[str] = Query(None),
    layers: Optional[int] = Query(None),
    uploader: Optional[str] = Query(None),
    force_min: Optional[float] = Query(None),
    force_max: Optional[float] = Query(None),
    pressure_min: Optional[float] = Query(None),
    pressure_max: Optional[float] = Query(None),
    temp_min: Optional[float] = Query(None),
    temp_max: Optional[float] = Query(None),
    angle_min: Optional[float] = Query(None),
    angle_max: Optional[float] = Query(None),
    result_status: Optional[str] = Query(
        None, alias="status",
        description="Filter by status (published, validated, etc.)"),
    limit: int = Query(100, ge=1, le=10000),
    order_by: str = Query("uploaded_at DESC"),
    viewer_name: Optional[str] = Depends(optional_api_key),
    db=Depends(get_db),
):
    """Query the shared database with optional filters.

    By default only ``published`` results are returned for unauthenticated
    queries. Pass ``status`` explicitly to override.
    """
    def _span(lo, hi):
        return (lo, hi) if lo is not None and hi is not None else None

    df = db.query(
        material=material,
        family=family,
        model=model,
        set_label=set_label,
        potential_type=potential_type,
        layers=layers,
        uploader=uploader,
        force_range=_span(force_min, force_max),
        pressure_range=_span(pressure_min, pressure_max),
        temperature_range=_span(temp_min, temp_max),
        angle_range=_span(angle_min, angle_max),
        limit=limit,
        order_by=order_by,
    )

    df = _filter_visible_rows(df, viewer_name, result_status)
    results = [_row_to_response(row) for _, row in df.iterrows()]
    return QueryResponse(count=len(results), results=results)


@app.get("/results/{result_id}", response_model=ResultResponse)
def get_result(result_id: int,
                viewer_name: Optional[str] = Depends(optional_api_key),
                db=Depends(get_db)):
    """Retrieve a single result by ID."""
    df = _filter_visible_rows(db.query(limit=None), viewer_name)
    match = df[df['id'] == result_id]
    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found",
        )
    row = match.iloc[0].to_dict()
    return _row_to_response(row)


@app.get("/statistics", response_model=StatisticsResponse)
def get_statistics(viewer_name: Optional[str] = Depends(optional_api_key),
                   db=Depends(get_db)):
    """Return aggregate statistics about the dataset."""
    df = _filter_visible_rows(db.query(limit=None), viewer_name)
    return _statistics_from_rows(df)


@app.get("/materials", response_model=MaterialsResponse)
def list_materials(viewer_name: Optional[str] = Depends(optional_api_key),
                   db=Depends(get_db)):
    """List distinct material names in the database."""
    df = _filter_visible_rows(db.query(limit=None), viewer_name)
    if df.empty:
        return MaterialsResponse(materials=[])
    materials = sorted(df['material'].dropna().unique().tolist())
    return MaterialsResponse(materials=materials)


@app.get("/conditions", response_model=ConditionsResponse)
def get_conditions(viewer_name: Optional[str] = Depends(optional_api_key),
                   db=Depends(get_db)):
    """Return parameter ranges available in the dataset."""
    df = _filter_visible_rows(db.query(limit=None), viewer_name)
    if df.empty:
        return ConditionsResponse()

    def _float_range(col: str) -> Optional[FloatRange]:
        s = df[col].dropna()
        if s.empty:
            return None
        lo, hi = s.min(), s.max()
        return {"min": float(lo), "max": float(hi)}

    def _int_range(col: str) -> Optional[IntRange]:
        s = df[col].dropna()
        if s.empty:
            return None
        lo, hi = s.min(), s.max()
        return {"min": int(lo), "max": int(hi)}

    def _counts(col: str) -> Optional[Dict[str, int]]:
        if col not in df.columns:
            return None
        series = df[col].dropna()
        return None if series.empty else {
            str(k): int(v) for k, v in series.value_counts().items()}

    return ConditionsResponse(
        force_nN=_float_range('force_nN'),
        temperature=_float_range('temperature'),
        scan_angle=_float_range('scan_angle'),
        pressure_gpa=_float_range('pressure_gpa'),
        layers=_int_range('layers'),
        family=_counts('family'),
        model=_counts('model'),
    )


# ---------------------------------------------------------------------------
# Authenticated endpoints (require X-API-Key header)
# ---------------------------------------------------------------------------


@app.post("/results", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def stage_result(
    body: ResultCreate,
    user_name: str = Depends(require_api_key),
    db=Depends(get_db),
):
    """Submit a new simulation result.

    Requires a valid API key (``X-API-Key`` header). The result is uploaded
    as ``staged``, automatically validated, and published when it passes
    validation.

    A duplicate ``time_series_hash`` (the same result uploaded twice) is not
    an upload failure — ``FrictionDB.upload_result`` returns ``None`` for it
    (``ON CONFLICT DO NOTHING``) and this raises a clean 409 rather than
    letting a non-optional ``UploadResponse.id`` crash on ``None``.
    """
    row_id = db.upload_result(
        uploader=user_name,
        status='staged',
        **body.model_dump(exclude_none=True),
    )
    if row_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate result: time_series_hash already exists",
        )
    validation = db.validate_staged(row_id)
    if validation.is_valid:
        db.publish(row_id)
        return UploadResponse(id=row_id, status='published')
    return UploadResponse(id=row_id, status='rejected')


@app.post("/results/{result_id}/validate", response_model=ValidationResponse)
def validate_result(
    result_id: int,
    _user_name: str = Depends(require_api_key),
    db=Depends(get_db),
):
    """Run automated validation on a staged result."""
    vr = db.validate_staged(result_id)
    return ValidationResponse(
        is_valid=vr.is_valid,
        errors=vr.errors,
        warnings=vr.warnings,
    )


@app.post("/results/{result_id}/publish")
def publish_result(
    result_id: int,
    _user_name: str = Depends(require_api_key),
    db=Depends(get_db),
):
    """Promote a validated result to published."""
    ok = db.publish(result_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found or already published",
        )
    return {"id": result_id, "status": "published"}


@app.post("/results/{result_id}/reject")
def reject_result(
    result_id: int,
    body: RejectRequest,
    _user_name: str = Depends(require_api_key),
    db=Depends(get_db),
):
    """Reject a staged or validated result."""
    ok = db.reject(result_id, reason=body.reason)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found",
        )
    return {"id": result_id, "status": "rejected"}
