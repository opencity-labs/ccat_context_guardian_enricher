import re
import requests
import json
from typing import Any, Dict, Optional
from cat.log import log
from cat.convo.messages import CatMessage

def transcribe_with_gemini(audio_data_uri: str, api_key: str) -> Dict[str, Any]:
    """
    Transcribe audio using Gemini API.
    Returns a dict with 'text' and 'usage' keys.
    """
    if not api_key:
        log.error(json.dumps({
            "event": "audio_transcription_error",
            "reason": "Gemini API key not found in settings"
        }))
        return {"text": "", "usage": None}

    match = re.match(r'data:(audio/[a-zA-Z0-9.-]+);base64,(.+)', audio_data_uri)
    if not match:
        log.error(json.dumps({
            "event": "audio_transcription_error",
            "reason": "Invalid audio data URI format"
        }))
        return {"text": "", "usage": None}
    
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
        
        usage = result.get("usageMetadata")
        
        if "candidates" in result and result["candidates"]:
            content = result["candidates"][0]["content"]
            parts = content.get("parts", [])
            text = "".join([p.get("text", "") for p in parts])
            return {"text": text.strip(), "usage": usage}
        else:
            log.error(json.dumps({
                "event": "audio_transcription_error",
                "reason": "Gemini API returned no candidates",
                "api_result": result
            }))
            return {"text": "", "usage": None}
            
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

        return {"text": "", "usage": None}

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

    result = transcribe_with_gemini(audio_data_uri, api_key)
    transcription = result.get("text", "")
    usage = result.get("usage")
    
    if transcription:
        # Log usage if available
        if usage:
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
            log.info(json.dumps({
                "event": "audio_transcription_success",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }))
        else:
            log.info(json.dumps({
                "event": "audio_transcription_success",
                "usage": "not available"
            }))
        
        cat.send_chat_message(CatMessage(user_id=cat.user_id, who="Human", text=transcription))
        return transcription
    else:
        log.warning(json.dumps({
            "event": "audio_transcription_warning",
            "reason": "Transcription failed or returned empty"
        }))
        return None
