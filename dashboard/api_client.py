"""Small wrapper around the HinSafe FastAPI backend."""

from typing import Any
import requests


HEALTH_TIMEOUT = 5        # seconds
FAST_TIMEOUT = 60         # prediction without SHAP
EXPLAIN_TIMEOUT = 180     # prediction with SHAP (slow on CPU)

CANNOT_REACH = "Cannot reach the API. Is the server running, and is the URL correct?"
PREDICTION_FIELDS = {
    "comment",
    "label",
    "severity_score",
    "severity_band",
    "categories",
    "flagged_words",
}


def check_health(base_url: str) -> tuple[dict[str, Any] | None, str | None]:
    """Ask the API whether it is up and the model is loaded."""
    try:
        response = requests.get(f"{base_url}/health", timeout=HEALTH_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except ValueError:
        return None, "The API returned an invalid health-check response."
    except requests.exceptions.ConnectionError:
        return None, CANNOT_REACH
    except requests.exceptions.Timeout:
        return None, "The API did not respond in time."
    except requests.exceptions.RequestException as error:
        return None, f"Health check failed: {error}"

    if not isinstance(payload, dict) or not isinstance(
        payload.get("model_loaded"), bool
    ):
        return None, "The API returned an unexpected health-check response."

    return payload, None


def predict_comment(
    base_url: str,
    text: str,
    explain: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send one comment to POST /predict."""
    timeout = EXPLAIN_TIMEOUT if explain else FAST_TIMEOUT

    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"text": text, "explain": explain},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        return None, CANNOT_REACH
    except requests.exceptions.Timeout:
        return None, "The API took too long to respond. SHAP is slow on CPU, try turning it off."
    except requests.exceptions.RequestException as error:
        return None, f"Request failed: {error}"

    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            return None, "The API returned an invalid prediction response."

        if isinstance(payload, dict) and PREDICTION_FIELDS.issubset(payload):
            return payload, None
        return None, "The API returned an unexpected prediction response."

    return None, f"API error {response.status_code}: {read_error_detail(response)}"


def read_error_detail(response: requests.Response) -> str:
    """Pull a readable message out of an API error response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text

    if not isinstance(payload, dict):
        return response.text

    detail = payload.get("detail", response.text)

    # Validation errors arrive as a list of dictionaries
    if isinstance(detail, list):
        return "; ".join(
            item.get("msg", str(item)) if isinstance(item, dict) else str(item)
            for item in detail
        )

    message = str(detail).strip()
    return message or "The API did not provide error details."
