import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
from cat.log import log
import requests
import json
from typing import Any
import tiktoken

_TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Return token count for text using tiktoken if available, fall back to len(text)."""
    if not _TIKTOKEN_ENCODING:
        return len(text)
    try:
        return len(_TIKTOKEN_ENCODING.encode(text))
    except Exception:
        return len(text)


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
    nlp_model.add_pipe("language_detector", last=True)

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
    # log.warning(
    #     f"Detected languages - Text 1: {lang1['language']} (confidence: {lang1['score']}), Text 2: {lang2['language']} (confidence: {lang2['score']})"
    # )
    return lang1["language"] == lang2["language"]


_STARTOF_TEXT_TAG = "<|startoftext|>"
_ENDOF_TEXT_TAG = "<|endoftext|>"


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

    api_key = settings.get("google_api_key", "")

    # 2. Prepare Prompt
    prompt = f"""Act as a translation engine. You will be provided with two texts, your task is to compare the language of the "Reference Text" with the "Text to Translate."
- Maintain the original tone and formatting.
- Output the result between {_STARTOF_TEXT_TAG} and {_ENDOF_TEXT_TAG}.
- No explanations, headers, or notes.

1. If both texts are in the same language: the output will be empty, so just: {_STARTOF_TEXT_TAG}{_ENDOF_TEXT_TAG};
2. If the two languages differ: the output will be ONLY the translation of the "Text to Translate" into the language used in the "Reference Text", so {_STARTOF_TEXT_TAG}TRANSLATED TEXT{_ENDOF_TEXT_TAG}

Reference Text for language: "{reference_text}"
Text to Translate: "{text_to_translate}" """

    # 3. Call Gemini API (if key exists)
    if api_key:
        # Try Gemini 2.5 Flash Lite first, fallback to 2.0 Flash Lite on 503
        models_to_try = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

            payload = {"contents": [{"parts": [{"text": prompt}]}]}

            try:
                response = requests.post(url, json=payload, timeout=30)

                if response.status_code == 503:
                    # Check if it's the high demand error
                    try:
                        error_data = response.json()
                        if (
                            error_data.get("error", {}).get("status") == "UNAVAILABLE"
                            and "high demand"
                            in error_data.get("error", {}).get("message", "").lower()
                        ):
                            # log.warning(f"Model {model_name} experiencing high demand, trying next model...")
                            continue  # Try next model
                    except:
                        pass

                    # If not the specific high demand error, return original text
                    log.error(
                        json.dumps(
                            {
                                "event": "translation_error",
                                "status_code": response.status_code,
                                "response": response.text,
                            }
                        )
                    )
                    return text_to_translate

                if response.status_code != 200:
                    log.error(
                        json.dumps(
                            {
                                "event": "translation_error",
                                "status_code": response.status_code,
                                "response": response.text,
                            }
                        )
                    )
                    return text_to_translate

                data = response.json()

                # Extract Content
                if "candidates" in data and data["candidates"]:
                    content = data["candidates"][0]["content"]
                    parts = content.get("parts", [])
                    raw_text = "".join([p.get("text", "") for p in parts]).strip()

                    # Parse the translated text between {_STARTOF_TEXT_TAG} and {_ENDOF_TEXT_TAG}
                    if _STARTOF_TEXT_TAG in raw_text and _ENDOF_TEXT_TAG in raw_text:
                        start_idx = raw_text.find(_STARTOF_TEXT_TAG) + len(
                            _STARTOF_TEXT_TAG
                        )
                        end_idx = raw_text.find(_ENDOF_TEXT_TAG)
                        translated_text = raw_text[start_idx:end_idx].strip()
                        if translated_text == "":
                            # If translation is empty, the model detected same language and returned empty translation, so we fallback to original text
                            translated_text = text_to_translate
                        log.info(
                            json.dumps(
                                {
                                    "event": "translation_success",
                                    "model": model_name,
                                    "input_length": _count_tokens(text_to_translate),
                                    "output_length": _count_tokens(translated_text),
                                }
                            )
                        )
                    else:
                        log.warning(
                            json.dumps(
                                {
                                    "event": "translation_warning",
                                    "reason": "Response does not contain expected tags",
                                    "raw_response": raw_text,
                                }
                            )
                        )
                        translated_text = text_to_translate.split("current time")[
                            0
                        ].strip()  # Fallback to original if format is unexpected

                    return translated_text

                log.warning(
                    json.dumps(
                        {
                            "event": "translation_warning",
                            "reason": "API returned no candidates",
                            "response": data,
                        }
                    )
                )
                return text_to_translate

            except Exception as e:
                log.error(
                    json.dumps(
                        {
                            "event": "translation_exception",
                            "model": model_name,
                            "error": str(e),
                        }
                    )
                )
                if (
                    model_name == models_to_try[-1]
                ):  # If this is the last model, return original
                    return text_to_translate
                else:
                    log.warning(
                        json.dumps(
                            {
                                "event": "translation_retry",
                                "reason": f"Exception with {model_name}, trying next model",
                            }
                        )
                    )
                    continue  # Try next model

    log.warning(
        json.dumps(
            {"event": "translation_skipped", "reason": "No Google API Key found"}
        )
    )
    return text_to_translate
