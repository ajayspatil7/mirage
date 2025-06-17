"""
Mirage ML Engine - FastAPI Service
Production-grade synthetic data generation API optimized for Apple Silicon M4 Pro.
"""

import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from loguru import logger

from .core.config import config, settings, get_system_info
from .core.exceptions import (
    MirageException,
    APIError,
    ValidationError,
    ConfigurationError,
    SystemRequirementError
)


# Global application state
app_state = {
    "startup_time": None,
    "health_status": "starting",
    "system_info": None,
    "request_count": 0,
    "error_count": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown.
    """
    # Startup
    logger.info("🔥 Starting Mirage ML Engine...")
    
    try:
        # Initialize system information
        app_state["system_info"] = get_system_info()
        app_state["startup_time"] = time.time()
        
        # Validate system requirements
        await validate_system_requirements()
        
        # Initialize ML components (if needed)
        await initialize_ml_components()
        
        app_state["health_status"] = "healthy"
        logger.info("✅ Mirage ML Engine started successfully")
        logger.info(f"🖥️  Running on device: {config.device}")
        logger.info(f"🚀 API available at http://{settings.host}:{settings.port}")
        
    except Exception as e:
        app_state["health_status"] = "error"
        logger.error(f"❌ Failed to start Mirage ML Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Mirage ML Engine...")
    app_state["health_status"] = "shutdown"
    logger.info("✅ Mirage ML Engine shut down successfully")


async def validate_system_requirements():
    """Validate that all system requirements are met."""
    try:
        # Check PyTorch and device availability
        import torch
        if not config.system.mps_available and settings.use_mps:
            logger.warning("⚠️  MPS not available, falling back to CPU")
        
        # Check storage paths
        for name, path in config.storage_paths.items():
            if not path.exists():
                raise SystemRequirementError(
                    f"Storage path does not exist: {path}",
                    requirement=f"storage_{name}"
                )
        
        logger.info("✅ System requirements validated")
        
    except Exception as e:
        raise SystemRequirementError(
            f"System validation failed: {e}",
            original_exception=e
        )


async def initialize_ml_components():
    """Initialize ML components and model registry."""
    try:
        # Initialize model registry (when we implement it)
        # Initialize job queue (when we implement it)
        logger.info("✅ ML components initialized")
        
    except Exception as e:
        raise ConfigurationError(
            f"ML component initialization failed: {e}",
            original_exception=e
        )


# Create FastAPI application
app = FastAPI(
    title="Mirage ML Engine",
    description="Production-grade synthetic data generation API optimized for Apple Silicon M4 Pro",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# =============================================================================
# Middleware Configuration
# =============================================================================

# CORS Middleware - Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted Host Middleware - Security
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.syntheta.ai"]
    )


# Request logging and metrics middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Log requests and track metrics."""
    start_time = time.time()
    app_state["request_count"] += 1
    
    # Log request
    logger.info(f"📨 {request.method} {request.url.path}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
        
        return response
        
    except Exception as e:
        # Log error
        app_state["error_count"] += 1
        duration = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - Error ({duration:.3f}s): {e}")
        raise


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(MirageException)
async def mirage_exception_handler(request: Request, exc: MirageException):
    """Handle all Mirage-specific exceptions."""
    logger.error(f"🚨 Mirage Exception: {exc.error_code} - {exc.message}")
    
    # Determine HTTP status code
    if isinstance(exc, ValidationError):
        status_code = exc.status_code
    elif isinstance(exc, APIError):
        status_code = exc.status_code
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"🚨 HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    app_state["error_count"] += 1
    
    # Log the full traceback for debugging
    logger.error(f"🚨 Unhandled Exception: {type(exc).__name__}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Return generic error to client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "details": {
                "type": type(exc).__name__,
                "message": str(exc) if settings.debug else "Internal server error"
            }
        }
    )


# =============================================================================
# Core API Routes
# =============================================================================

@app.get("/", summary="Root endpoint", tags=["System"])
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Mirage ML Engine",
        "version": "0.1.0",
        "description": "Production-grade synthetic data generation API",
        "status": app_state["health_status"],
        "uptime_seconds": time.time() - app_state["startup_time"] if app_state["startup_time"] else 0,
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", summary="Health check", tags=["System"])
async def health_check():
    """Comprehensive health check endpoint."""
    health_data = {
        "status": app_state["health_status"],
        "timestamp": time.time(),
        "uptime_seconds": time.time() - app_state["startup_time"] if app_state["startup_time"] else 0,
        "version": "0.1.0",
        "environment": settings.environment,
        "metrics": {
            "total_requests": app_state["request_count"],
            "total_errors": app_state["error_count"],
            "error_rate": app_state["error_count"] / max(app_state["request_count"], 1) * 100
        }
    }
    
    # Add system info if available
    if app_state["system_info"]:
        health_data["system"] = app_state["system_info"]
    
    # Determine HTTP status based on health
    if app_state["health_status"] == "healthy":
        return health_data
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_data
        )


@app.get("/system/info", summary="System information", tags=["System"])
async def system_info():
    """Get detailed system information."""
    if not app_state["system_info"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System information not available"
        )
    
    return {
        "system_info": app_state["system_info"],
        "config": {
            "environment": settings.environment,
            "debug": settings.debug,
            "device": str(config.device),
            "storage_paths": {k: str(v) for k, v in config.storage_paths.items()},
            "max_concurrent_jobs": settings.max_concurrent_jobs
        }
    }


@app.get("/system/metrics", summary="System metrics", tags=["System"])
async def system_metrics():
    """Get system performance metrics."""
    import psutil
    import torch
    
    metrics = {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - app_state["startup_time"] if app_state["startup_time"] else 0,
        "requests": {
            "total": app_state["request_count"],
            "errors": app_state["error_count"],
            "error_rate_percent": app_state["error_count"] / max(app_state["request_count"], 1) * 100
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
    
    # Add GPU metrics if MPS is available
    if config.system.mps_available:
        try:
            # Basic MPS info (limited metrics available on MPS)
            metrics["mps"] = {
                "available": True,
                "device": str(config.device)
            }
        except Exception:
            metrics["mps"] = {"available": False}
    
    return metrics


# =============================================================================
# API Documentation Customization
# =============================================================================

def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Mirage ML Engine",
        version="0.1.0",
        description="""
# Mirage ML Engine API

Production-grade synthetic data generation API optimized for Apple Silicon M4 Pro.

## Features

- 🚀 **Fast**: Optimized for Apple Silicon M4 Pro with MPS acceleration
- 🏥 **Healthcare-Ready**: HIPAA-compliant synthetic medical data generation  
- 📊 **Multiple Models**: CTGAN, TVAE, CopulaGAN support
- 🔧 **Production-Grade**: Async job processing, metrics, monitoring
- 🎯 **API-First**: Complete REST API with OpenAPI documentation

## Authentication

Currently no authentication required for development. Production deployment will include API key authentication.

## Rate Limits

- Development: No rate limits
- Production: 100 requests per minute per IP

## Support

- Documentation: `/docs`
- Health Check: `/health`
- System Info: `/system/info`
        """,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://syntheta.ai/logo.svg"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =============================================================================
# Development Server
# =============================================================================

def start_server():
    """Start the development server."""
    logger.info("🚀 Starting Mirage ML Engine development server...")
    
    uvicorn.run(
        "mirage.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=settings.debug
    )


if __name__ == "__main__":
    start_server()