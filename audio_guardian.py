import re
import requests
import json
from typing import Any, Dict, Optional
from cat.log import log
from cat.convo.messages import CatMessage

def transcribe_with_gemini(audio_data_uri: str, api_key: str) -> str:
    """
    Transcribe audio using Gemini API.
    """
    if not api_key:
        log.error(json.dumps({
            "event": "audio_transcription_error",
            "reason": "Gemini API key not found in settings"
        }))
        return ""

    match = re.match(r'data:(audio/[a-zA-Z0-9.-]+);base64,(.+)', audio_data_uri)
    if not match:
        log.error(json.dumps({
            "event": "audio_transcription_error",
            "reason": "Invalid audio data URI format"
        }))
        return ""
    
    mime_type = match.group(1)
    base64_data = match.group(2)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": "Transcribe the following audio exactly as it is spoken. Do not add any description or timestamp."},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_data
                    }
                }
            ]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "candidates" in result and result["candidates"]:
            content = result["candidates"][0]["content"]
            parts = content.get("parts", [])
            text = "".join([p.get("text", "") for p in parts])
            return text.strip()
        else:
            log.error(json.dumps({
                "event": "audio_transcription_error",
                "reason": "Gemini API returned no candidates",
                "api_result": result
            }))
            return ""
            
    except Exception as e:
        log.error(json.dumps({
            "event": "audio_transcription_error",
            "reason": str(e)
        }))
        if 'response' in locals() and hasattr(response, 'text'):
            log.error(json.dumps({
                "event": "audio_transcription_api_response",
                "response_text": response.text
            }))
            
            # Fallback to list models if 404 to help debugging
            if hasattr(response, 'status_code') and response.status_code == 404:
                 try:
                     list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                     list_resp = requests.get(list_url, timeout=10)
                     if list_resp.status_code == 200:
                         models = [m['name'] for m in list_resp.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
                         log.error(json.dumps({
                             "event": "available_models",
                             "models": models
                         }))
                 except:
                     pass

        return ""

def handle_audio_transcription(audio_data_uri: str, cat: Any) -> Optional[str]:
    """
    Handle audio transcription using Gemini API.

    Args:
        audio_data_uri: The audio data URI to transcribe.
        cat: The StrayCat instance to send messages.

    Returns:
        The transcribed text if successful, None otherwise.
    """
    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    api_key = settings.get('google_api_key', '')

    if not api_key:
        log.warning(json.dumps({
            "event": "audio_transcription_warning",
            "reason": "Google API Key missing"
        }))
        return None

    transcription = transcribe_with_gemini(audio_data_uri, api_key)
    if transcription:
        # log.info(f"Audio transcribed: {transcription}")
        cat.send_chat_message(CatMessage(user_id=cat.user_id, who="Human", text=transcription))
        return transcription
    else:
        log.warning(json.dumps({
            "event": "audio_transcription_warning",
            "reason": "Transcription failed or returned empty"
        }))
        return None
