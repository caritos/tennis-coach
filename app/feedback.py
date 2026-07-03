"""Turns structured scoring findings into natural-language coaching feedback
using a locally running Ollama model. The LLM only rephrases findings that
were already decided by the scoring model — it does not judge technique itself."""
import requests

from .scoring import ScoreResult

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaUnavailableError(RuntimeError):
    pass


def _build_prompt(result: ScoreResult) -> str:
    if not result.findings:
        findings_text = "No technique issues were flagged — this swing is close to the reference norm."
    else:
        findings_text = "\n".join(
            f"- {f.feature}: measured {f.value:.1f}, reference average {f.reference_mean:.1f} "
            f"({f.severity} deviation)"
            for f in result.findings
        )

    return (
        "You are a supportive tennis coach reviewing a student's forehand. "
        "Below are technique measurements compared to a reference model of good "
        "forehand form. Write 2-4 short paragraphs of encouraging, specific "
        "coaching feedback: mention what's working, explain each flagged issue "
        "in plain language, and suggest one concrete drill or cue per issue. "
        "Do not invent technique issues beyond what is listed below.\n\n"
        f"Measurements:\n{findings_text}\n"
    )


def generate_feedback(result: ScoreResult, model: str = DEFAULT_MODEL) -> str:
    prompt = _build_prompt(result)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at localhost:11434 — is `ollama serve` running "
            f"and has `{model}` been pulled?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(
            f"Ollama at localhost:11434 timed out generating a response — the local "
            f"model may be slow or stuck. Try again, or check `ollama serve` logs."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise OllamaUnavailableError(
            f"Ollama returned an error for model `{model}` — has it been pulled? ({exc})"
        ) from exc

    return response.json()["response"]
