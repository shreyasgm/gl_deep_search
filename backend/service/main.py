"""
Main FastAPI application entry point.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.service.routes import router as publications_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app instance
app = FastAPI(
    title="Publication Tracking API",
    description="API for tracking publication metadata and ETL pipeline status",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(publications_router, prefix="/api/v1")


@app.get("/api/health", tags=["health"])
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.service.main:app", host="0.0.0.0", port=8000, reload=True)
