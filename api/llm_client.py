import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_prompt(
    score: float,
    category: str,
    raw_features: dict,
    top_contributors: list[dict],
) -> str:
    """
    Builds a structured, data-rich prompt so the LLM explains
    the specific prediction — not generic advice.
    """
    contributors_text = "\n".join([
        f"  - {c['feature']}: value={c['value']:.2f}, "
        f"SHAP impact={c['shap_value']:+.2f} on score"
        for c in top_contributors
    ])

    prompt = f"""You are a workplace productivity analyst. A machine learning model has analysed a user's lifestyle and digital habits and predicted their Work Productivity Score.

Predicted Score: {score:.1f} / 100 ({category} productivity)

User's key metrics:
  - Sleep: {raw_features['Sleep_Hours']} hours/night (deficit: {max(0, 8 - raw_features['Sleep_Hours']):.1f}h)
  - Stress Level: {raw_features['Stress_Level']}/10
  - Daily Phone Use: {raw_features['Daily_Phone_Hours']} hours
  - Social Media: {raw_features['Social_Media_Hours']} hours
  - Caffeine: {raw_features['Caffeine_Intake_Cups']} cups/day
  - Weekend Screen Time: {raw_features['Weekend_Screen_Time_Hours']} hours
  - Apps Used Daily: {raw_features['App_Usage_Count']}
  - Occupation: {raw_features['Occupation']}

Top 5 model features driving this prediction (SHAP values — positive means boosting productivity, negative means reducing it):
{contributors_text}

In 4–5 sentences:
1. Explain what is primarily driving this {"low" if score < 50 else "high"} productivity score, referencing the specific numbers above.
2. Identify the single biggest factor the user should address.
3. Give 2 concrete, actionable recommendations tailored to their profile.If they have good numbers,then dont nitpick and praise them for their lifestyle.
Be direct and specific. Do not give generic wellness advice. Reference the actual numbers."""

    return prompt


async def check_ollama_health(ollama_base_url: str, model: str = None) -> bool:
    """Returns True only if Ollama is reachable AND model is loaded."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if model:
                # Actually load the model by sending a dummy request
                r = await client.post(
                    f"{ollama_base_url}/api/generate",
                    json={"model": model, "prompt": "hi", "stream": False}
                )
            else:
                r = await client.get(f"{ollama_base_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def get_llm_insight_sync(
    score: float,
    category: str,
    raw_features: dict,
    top_contributors: list[dict],
    ollama_base_url: str,
    model: str = "ministral-3:3b",
    timeout: float = 60.0,
) -> str:
    """
    Sync version for Celery workers.
    """
    prompt = build_prompt(score, category, raw_features, top_contributors)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.8,
            "num_predict": 300,
        }
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama.")
        return "⚠️ Cannot connect to LLM."

    except httpx.TimeoutException:
        logger.error("Ollama timeout.")
        return "⚠️ LLM timed out."

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"⚠️ LLM error: {str(e)}"