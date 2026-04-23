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


def create_app(db=None, *, profile: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db: An existing :class:`~src.data.database.FrictionDB` instance.
            If ``None``, one is created from the given *profile*.
        profile: Database settings profile (``'local'`` or ``'central'``).
            Ignored if *db* is provided.

    Returns:
        Configured :class:`FastAPI` application.
    """
    if db is None:
        from ..data.database import db_from_profile  # noqa: PLC0415
        db = db_from_profile(profile)

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
    """Schema for staging a new result via the API."""
    material: str
    simulation_type: str = 'afm'
    layers: Optional[int] = None
    size_x: Optional[float] = None
    size_y: Optional[float] = None
    stack_type: Optional[str] = None
    force_nN: Optional[float] = None
    pressure_gpa: Optional[float] = None
    scan_angle: Optional[float] = None
    scan_speed: Optional[float] = None
    temperature: Optional[float] = None
    tip_material: Optional[str] = None
    tip_radius: Optional[float] = None
    substrate_material: Optional[str] = None
    substrate_amorphous: Optional[bool] = None
    potential_type: Optional[str] = None
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
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    data_url: Optional[str] = None


class ResultResponse(BaseModel):
    """Schema for a single result in API responses."""
    id: int
    uploaded_at: Optional[str] = None
    uploader: Optional[str] = None
    material: Optional[str] = None
    simulation_type: Optional[str] = None
    layers: Optional[int] = None
    size_x: Optional[float] = None
    size_y: Optional[float] = None
    stack_type: Optional[str] = None
    force_nN: Optional[float] = None
    pressure_gpa: Optional[float] = None
    scan_angle: Optional[float] = None
    scan_speed: Optional[float] = None
    temperature: Optional[float] = None
    tip_material: Optional[str] = None
    tip_radius: Optional[float] = None
    substrate_material: Optional[str] = None
    substrate_amorphous: Optional[bool] = None
    potential_type: Optional[str] = None
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
    status: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
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
    """Parameter ranges in the dataset."""
    force_nN: Optional[Dict[str, float]] = None
    temperature: Optional[Dict[str, float]] = None
    scan_angle: Optional[Dict[str, float]] = None
    pressure_gpa: Optional[Dict[str, float]] = None
    layers: Optional[Dict[str, int]] = None


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
    id: int
    status: str


# ---------------------------------------------------------------------------
# Helper to convert DataFrame row to ResultResponse
# ---------------------------------------------------------------------------


def _row_to_response(row: dict) -> ResultResponse:
    """Convert a pandas row dict to a ResultResponse."""
    # Handle timestamp serialisation
    uploaded = row.get('uploaded_at')
    if uploaded is not None:
        uploaded = str(uploaded)
    return ResultResponse(
        id=row['id'],
        uploaded_at=uploaded,
        uploader=row.get('uploader'),
        material=row.get('material'),
        simulation_type=row.get('simulation_type'),
        layers=row.get('layers'),
        size_x=row.get('size_x'),
        size_y=row.get('size_y'),
        stack_type=row.get('stack_type'),
        force_nN=row.get('force_nN'),
        pressure_gpa=row.get('pressure_gpa'),
        scan_angle=row.get('scan_angle'),
        scan_speed=row.get('scan_speed'),
        temperature=row.get('temperature'),
        tip_material=row.get('tip_material'),
        tip_radius=row.get('tip_radius'),
        substrate_material=row.get('substrate_material'),
        substrate_amorphous=row.get('substrate_amorphous'),
        potential_type=row.get('potential_type'),
        mean_cof=row.get('mean_cof'),
        std_cof=row.get('std_cof'),
        mean_lf=row.get('mean_lf'),
        std_lf=row.get('std_lf'),
        mean_nf=row.get('mean_nf'),
        std_nf=row.get('std_nf'),
        mean_lfx=row.get('mean_lfx'),
        std_lfx=row.get('std_lfx'),
        mean_lfy=row.get('mean_lfy'),
        std_lfy=row.get('std_lfy'),
        ntimesteps=row.get('ntimesteps'),
        time_series_hash=row.get('time_series_hash'),
        is_complete=row.get('is_complete'),
        status=row.get('status'),
        notes=row.get('notes'),
        metadata=row.get('metadata'),
        data_url=row.get('data_url'),
    )


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
    simulation_types = df['simulation_type'].dropna().value_counts().to_dict()
    cof_values = df['mean_cof'].dropna()
    cof_global_mean = float(cof_values.mean()) if not cof_values.empty else None
    return StatisticsResponse(
        total_rows=len(df),
        by_material=materials,
        by_type=simulation_types,
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
def query_results(
    material: Optional[str] = Query(None),
    simulation_type: Optional[str] = Query(None),
    layers: Optional[int] = Query(None),
    force_min: Optional[float] = Query(None),
    force_max: Optional[float] = Query(None),
    temp_min: Optional[float] = Query(None),
    temp_max: Optional[float] = Query(None),
    angle_min: Optional[float] = Query(None),
    angle_max: Optional[float] = Query(None),
    uploader: Optional[str] = Query(None),
    result_status: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status (published, validated, etc.)",
    ),
    limit: int = Query(100, ge=1, le=10000),
    order_by: str = Query("uploaded_at DESC"),
    viewer_name: Optional[str] = Depends(optional_api_key),
    db=Depends(get_db),
):
    """Query the shared database with optional filters.

    By default only ``published`` results are returned for unauthenticated
    queries. Pass ``status`` explicitly to override.
    """
    force_range = None
    if force_min is not None and force_max is not None:
        force_range = (force_min, force_max)
    temp_range = None
    if temp_min is not None and temp_max is not None:
        temp_range = (temp_min, temp_max)
    angle_range = None
    if angle_min is not None and angle_max is not None:
        angle_range = (angle_min, angle_max)

    df = db.query(
        material=material,
        simulation_type=simulation_type,
        layers=layers,
        force_range=force_range,
        temperature_range=temp_range,
        angle_range=angle_range,
        uploader=uploader,
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

    return ConditionsResponse(
        force_nN=_float_range('force_nN'),
        temperature=_float_range('temperature'),
        scan_angle=_float_range('scan_angle'),
        pressure_gpa=_float_range('pressure_gpa'),
        layers=_int_range('layers'),
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
    """
    row_id = db.upload_result(
        uploader=user_name,
        status='staged',
        **body.model_dump(exclude_none=True),
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
