
from fastapi import APIRouter, Request

from schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
def health(request: Request) -> HealthResponse:
    """Return whether the API is responding and the model is ready."""

    model_loaded = getattr(request.app.state, "pipeline", None) is not None
    return HealthResponse(model_loaded=model_loaded)
