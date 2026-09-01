"""FastAPI application entry point for AdLens."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from adlens.api.routes import analyze, health
from adlens.utils.config import settings
from adlens.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings.work_path.mkdir(parents=True, exist_ok=True)
    logger.info("AdLens API starting up")
    logger.info("  ASR provider   : %s", settings.effective_asr_provider)
    logger.info("  Vision provider: %s", settings.effective_vision_provider)
    logger.info("  Work dir       : %s", settings.work_dir)
    logger.info("  OpenAI key     : %s", "SET" if settings.has_openai_key else "NOT SET (mock mode)")
    yield
    logger.info("AdLens API shutting down")


app = FastAPI(
    title="AdLens",
    description=(
        "Multimodal video advertisement segment detection system. "
        "Detects advertisement segments in YouTube videos and local files "
        "using ASR, OCR, visual classification, and scene detection."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permissive for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["health"])
app.include_router(analyze.router, tags=["analysis"])

# Serve the minimal web viewer
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(web_dir), html=True), name="viewer")

