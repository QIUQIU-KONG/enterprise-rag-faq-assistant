"""
FastAPI application entry point.
Configures CORS, lifespan, and mounts all routes.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router
from src.core.rag_engine import get_rag_engine
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize RAG engine on startup."""
    logger.info("Starting FAQ Knowledge Base API...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")

    # Initialize the RAG engine
    engine = get_rag_engine()
    engine.initialize()

    logger.info(f"API ready. Index size: {engine.index_size()} chunks")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title="FAQ Knowledge Base Assistant",
        description="Internal AI-powered FAQ bot for travel, visa, and project applications",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — restricted by default; configure ALLOWED_ORIGINS for your deployment
    origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://localhost:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # Mount routes
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )
