import re
import requests
import json
from typing import Any, Dict, Optional
from cat.log import log
from cat.convo.messages import CatMessage

_TRANSCRIPTION_MODEL = "gemini-2.5-pro"
_TRANSCRIPTION_PROMPT = "Transcribe the following audio exactly as it is spoken. Do not add any description or timestamp."


def _parse_audio_data_uri(audio_data_uri: str) -> Optional[Dict[str, str]]:
    """Parse a data URI into mime_type and base64 data."""
    match = re.match(r"data:(audio/[a-zA-Z0-9.-]+);base64,(.+)", audio_data_uri)
    if not match:
        log.error(
            json.dumps(
                {
                    "event": "audio_transcription_error",
                    "reason": "Invalid audio data URI format",
                }
            )
        )
        return None
    return {"mime_type": match.group(1), "data": match.group(2)}


def _build_payload(mime_type: str, base64_data: str) -> Dict:
    """Build the generateContent payload (same format for both Gemini and Vertex AI)."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _TRANSCRIPTION_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": base64_data}},
                ],
            }
        ]
    }


def _parse_generate_content_response(result: Dict, source: str) -> Dict[str, Any]:
    """Parse a generateContent API response into text and usage."""
    usage = result.get("usageMetadata")
    if "candidates" in result and result["candidates"]:
        content = result["candidates"][0]["content"]
        parts = content.get("parts", [])
        text = "".join([p.get("text", "") for p in parts])
        return {"text": text.strip(), "usage": usage}
    else:
        log.error(
            json.dumps(
                {
                    "event": "audio_transcription_error",
                    "reason": f"{source} API returned no candidates",
                    "api_result": result,
                }
            )
        )
        return {"text": "", "usage": None}


def _get_vertex_ai_config(cat: Any) -> Optional[Dict[str, str]]:
    """Check if Vertex AI is the selected LLM provider and return its config.

    Reads project and location directly from the instantiated ChatVertexAI
    object on cat._llm, because the Vertex AI provider's get_llm_from_config
    pops those keys from the TinyDB-cached dict during bootstrap.
    """
    try:
        from langchain_google_vertexai import ChatVertexAI

        llm = cat._llm
        if isinstance(llm, ChatVertexAI):
            return {
                "project_id": llm.project,
                "location": llm.location,
            }
    except ImportError:
        pass
    except Exception as e:
        log.warning(
            json.dumps({"event": "vertex_ai_config_check_failed", "error": str(e)})
        )
    return None


def _get_vertex_ai_access_token() -> Optional[str]:
    """Get an OAuth2 access token using Application Default Credentials."""
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except Exception as e:
        log.error(json.dumps({"event": "vertex_ai_auth_error", "error": str(e)}))
        return None


def transcribe_with_vertex_ai(
    audio_data_uri: str, project_id: str, location: str
) -> Dict[str, Any]:
    """Transcribe audio using Vertex AI Gemini API."""
    parsed = _parse_audio_data_uri(audio_data_uri)
    if not parsed:
        return {"text": "", "usage": None}

    access_token = _get_vertex_ai_access_token()
    if not access_token:
        return {"text": "", "usage": None}

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/{_TRANSCRIPTION_MODEL}:generateContent"
    )
    payload = _build_payload(parsed["mime_type"], parsed["data"])
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return _parse_generate_content_response(response.json(), "Vertex AI")
    except Exception as e:
        log.error(
            json.dumps(
                {
                    "event": "audio_transcription_error",
                    "source": "vertex_ai",
                    "reason": str(e),
                }
            )
        )
        if "response" in locals() and hasattr(response, "text"):
            log.error(
                json.dumps(
                    {
                        "event": "audio_transcription_api_response",
                        "source": "vertex_ai",
                        "response_text": response.text,
                    }
                )
            )
        return {"text": "", "usage": None}


def transcribe_with_gemini(audio_data_uri: str, api_key: str) -> Dict[str, Any]:
    """Transcribe audio using Gemini API with API key."""
    if not api_key:
        log.error(
            json.dumps(
                {
                    "event": "audio_transcription_error",
                    "reason": "Gemini API key not found in settings",
                }
            )
        )
        return {"text": "", "usage": None}

    parsed = _parse_audio_data_uri(audio_data_uri)
    if not parsed:
        return {"text": "", "usage": None}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_TRANSCRIPTION_MODEL}:generateContent?key={api_key}"
    payload = _build_payload(parsed["mime_type"], parsed["data"])

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return _parse_generate_content_response(response.json(), "Gemini")
    except Exception as e:
        log.error(
            json.dumps(
                {
                    "event": "audio_transcription_error",
                    "source": "gemini_api_key",
                    "reason": str(e),
                }
            )
        )
        if "response" in locals() and hasattr(response, "text"):
            log.error(
                json.dumps(
                    {
                        "event": "audio_transcription_api_response",
                        "source": "gemini_api_key",
                        "response_text": response.text,
                    }
                )
            )
        return {"text": "", "usage": None}


def handle_audio_transcription(audio_data_uri: str, cat: Any) -> Optional[str]:
    """
    Handle audio transcription.

    Uses Vertex AI credentials if the Vertex AI provider is selected,
    otherwise falls back to the Google API key from plugin settings.

    Args:
        audio_data_uri: The audio data URI to transcribe.
        cat: The StrayCat instance to send messages.

    Returns:
        The transcribed text if successful, None otherwise.
    """
    result = None

    # Try Vertex AI first if it's the selected LLM provider
    vertex_config = _get_vertex_ai_config(cat)
    if vertex_config:
        log.info(
            json.dumps({"event": "audio_transcription_attempt", "source": "vertex_ai"})
        )
        result = transcribe_with_vertex_ai(
            audio_data_uri,
            vertex_config["project_id"],
            vertex_config["location"],
        )

    # Fall back to Gemini API key if Vertex AI is not available or failed
    if not result or not result.get("text"):
        settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
        api_key = settings.get("google_api_key", "")
        if api_key:
            if vertex_config:
                log.warning(
                    json.dumps(
                        {
                            "event": "audio_transcription_fallback",
                            "reason": "Vertex AI failed, falling back to Gemini API key",
                        }
                    )
                )
            result = transcribe_with_gemini(audio_data_uri, api_key)
        elif not vertex_config:
            log.warning(
                json.dumps(
                    {
                        "event": "audio_transcription_warning",
                        "reason": "No Vertex AI provider and no Google API Key configured",
                    }
                )
            )
            return None

    transcription = result.get("text", "") if result else ""
    usage = result.get("usage") if result else None

    if transcription:
        if usage:
            log.info(
                json.dumps(
                    {
                        "event": "audio_transcription_success",
                        "input_tokens": usage.get("promptTokenCount", 0),
                        "output_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0),
                    }
                )
            )
        else:
            log.info(
                json.dumps(
                    {"event": "audio_transcription_success", "usage": "not available"}
                )
            )

        cat.send_chat_message(
            CatMessage(user_id=cat.user_id, who="Human", text=transcription)
        )
        return transcription
    else:
        log.warning(
            json.dumps(
                {
                    "event": "audio_transcription_warning",
                    "reason": "Transcription failed or returned empty",
                }
            )
        )
        return None
