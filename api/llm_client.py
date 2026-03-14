import logging
from groq import Groq
import os

logger = logging.getLogger(__name__)

def build_prompt(
    score: float,
    category: str,
    raw_features: dict,
    top_contributors: list[dict],
) -> str:
    contributors_text = "\n".join([
        f"  - {c['feature']}: value={c['value']:.2f}, "
        f"SHAP impact={c['shap_value']:+.2f} on score"
        for c in top_contributors
    ])
    prompt1 = f"""You are a workplace productivity analyst. A machine learning model has analysed a user's lifestyle and digital habits and predicted their Work Productivity Score.
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
                    In 2-3 sentences:
                    1. Explain what is primarily driving this {"low" if score < 50 else "high"} productivity score, referencing the specific numbers above.
                    2. Identify the single biggest factor the user should address.
                    3. Give 2 concrete, actionable recommendations tailored to their profile. If they have good numbers, then dont nitpick and praise them for their lifestyle.
                    Be direct and specific. Do not give generic wellness advice. Reference the actual numbers."""
    return prompt1


def get_llm_insight_sync(
    score: float,
    category: str,
    raw_features: dict,
    top_contributors: list[dict],
    **kwargs,  # absorbs any leftover ollama args gracefully
) -> str:
    """
    Sync version for Celery workers. Calls Groq API.
    """
    prompt = build_prompt(score, category, raw_features, top_contributors)
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return f"⚠️ LLM error: {str(e)}"


async def check_groq_health() -> bool:
    """Verifies Groq API key exists AND actually works."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,  # minimal — just checking connectivity
        )
        return True
    except Exception as e:
        logger.error(f"Groq health check failed: {e}")
        return False