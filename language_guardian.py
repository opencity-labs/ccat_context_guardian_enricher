import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
from cat.log import log
import requests
from typing import Any


def get_lang_detector(nlp, name):
    """Factory function for creating a language detector component."""
    return LanguageDetector(seed=42)


def _initialize_language_model():
    """Initialize and configure the spacy language model with language detection."""
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")

    Language.factory("language_detector", func=get_lang_detector)
    nlp_model.add_pipe('language_detector', last=True)
    
    return nlp_model


# Initialize the model once at module level
_nlp_model = _initialize_language_model()


def detect_language(text: str) -> str:
    """
    Detect the language of a given text.
    
    Args:
        text: The text to detect the language for.
        
    Returns:
        The detected language code (e.g., 'en', 'de', 'es', 'fr').
    """
    doc = _nlp_model(text)
    return doc._.language


def is_same_language(text1: str, text2: str) -> bool:
    """
    Check if two strings are in the same language.
    
    Args:
        text1: The first text string.
        text2: The second text string.
        
    Returns:
        True if both texts are in the same language, False otherwise.
    """
    lang1 = detect_language(text1)
    lang2 = detect_language(text2)
    return lang1["language"] == lang2["language"]


def translate_text(text_to_translate: str, reference_text: str, cat: Any) -> str:
    """
    Translates text_to_translate to the language of reference_text.
    """
    # 1. Get Settings & API Key
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
    except Exception:
        # Fallback if plugin logic fails (rare)
        return text_to_translate

    api_key = settings.get('google_api_key', '')

    # 2. Prepare Prompt
    prompt = f"""You are a helpful translator.
I will provide you with a reference text and a text to translate.
Your goal is to translate the "Text to translate" into the same language as the "Reference text".

Reference text: "{reference_text}"

Text to translate: "{text_to_translate}"

Only output the translated text, nothing else. Do not add explanations."""

    # 3. Call Gemini API (if key exists)
    if api_key:
        # Try Gemini 2.5 Flash Lite first, fallback to 2.0 Flash Lite on 503
        models_to_try = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }

            try:
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 503:
                    # Check if it's the high demand error
                    try:
                        error_data = response.json()
                        if (error_data.get("error", {}).get("status") == "UNAVAILABLE" and 
                            "high demand" in error_data.get("error", {}).get("message", "").lower()):
                            # log.warning(f"Model {model_name} experiencing high demand, trying next model...")
                            continue  # Try next model
                    except:
                        pass
                    
                    # If not the specific high demand error, return original text
                    log.error(f"Translation API error ({response.status_code}): {response.text}")
                    return text_to_translate
                
                if response.status_code != 200:
                    log.error(f"Translation API error ({response.status_code}): {response.text}")
                    return text_to_translate

                data = response.json()
                
                # Track Usage
                if "usageMetadata" in data:
                    usage = data["usageMetadata"]
                    input_tokens = usage.get("promptTokenCount", 0)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                    
                    # Store usage in cat instance (transient) for analytics to pick up
                    if not hasattr(cat, "translation_usage"):
                        cat.translation_usage = {"input": 0, "output": 0}
                    
                    cat.translation_usage["input"] += input_tokens
                    cat.translation_usage["output"] += output_tokens
                
                # Extract Content
                if "candidates" in data and data["candidates"]:
                    content = data["candidates"][0]["content"]
                    parts = content.get("parts", [])
                    translated_text = "".join([p.get("text", "") for p in parts]).strip()
                    
                    log.info(f"Translation successful with {model_name}. Input length: {len(text_to_translate)}, Output length: {len(translated_text)}")
                    
                    # Store usage in cat instance (transient) for analytics to pick up
                    if not hasattr(cat, "translation_usage"):
                        cat.translation_usage = {"input": 0, "output": 0}
                    
                    cat.translation_usage["input"] += input_tokens
                    cat.translation_usage["output"] += output_tokens
                    # log.info(f"[Language Guardian] Stored translation usage in cat: {cat.translation_usage}")
                    
                    return translated_text
                
                log.warning(f"Translation API returned no candidates. Response: {data}")
                return text_to_translate
                
            except Exception as e:
                log.error(f"Translation exception with {model_name}: {e}")
                if model_name == models_to_try[-1]:  # If this is the last model, return original
                    return text_to_translate
                else:
                    log.warning(f"Trying next model after exception with {model_name}...")
                    continue  # Try next model

    log.warning("Skipping translation: No Google API Key found.")
    return text_to_translate
