"""REST API for the FrictionSim2D shared database.

Provides a FastAPI application that exposes the shared PostgreSQL database
over HTTP, enabling remote querying and result submission.

Usage::

    # Development server
    FrictionSim2D api serve --port 8000

    # Or directly with the app factory (ensures DB is initialized)
    uvicorn src.api.server:create_app --factory --host 0.0.0.0 --port 8000
"""
