"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from adlens.utils.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return service health and configuration summary."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "providers": {
            "asr": settings.effective_asr_provider,
            "vision": settings.effective_vision_provider,
            "sampling": settings.sampling_strategy,
        },
        "openai_key_configured": settings.has_openai_key,
    }
