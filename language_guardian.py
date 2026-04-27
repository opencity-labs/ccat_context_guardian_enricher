import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
from cat.log import log
import json
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
    return lang1["language"] == lang2["language"]


_STARTOF_TEXT_TAG = "<|startoftext|>"
_ENDOF_TEXT_TAG = "<|endoftext|>"


def translate_text(
    text_to_translate: str, reference_text: str, cat: Any, current_message: str = ""
) -> str:
    """
    Translates text_to_translate to the language of reference_text
    using the LLM configured in the Cheshire Cat framework.
    """
    prompt = f"""Act as a translation engine. Determine the user's intended language and translate the "Text to Translate" if needed.
- Maintain the original tone and formatting.
- Output the result between {_STARTOF_TEXT_TAG} and {_ENDOF_TEXT_TAG}.
- No explanations, headers, or notes.

Determine the target language with this priority:
1. Use "Last User Message" as the primary language indicator.
2. If "Last User Message" is too short or ambiguous to identify a language clearly, infer it from "Conversation History".

Translation rules:
- If "Text to Translate" is already in the target language: output empty → {_STARTOF_TEXT_TAG}{_ENDOF_TEXT_TAG}
- Otherwise: output ONLY the translation into the target language → {_STARTOF_TEXT_TAG}TRANSLATED TEXT{_ENDOF_TEXT_TAG}

Last User Message: "{current_message}"
Conversation History: "{reference_text}"
Text to Translate: "{text_to_translate}" """

    try:
        raw_text = cat.llm(prompt)
    except Exception as e:
        log.error(
            json.dumps(
                {
                    "event": "translation_exception",
                    "error": str(e),
                }
            )
        )
        return text_to_translate

    # Parse the translated text between tags
    if _STARTOF_TEXT_TAG in raw_text and _ENDOF_TEXT_TAG in raw_text:
        start_idx = raw_text.find(_STARTOF_TEXT_TAG) + len(_STARTOF_TEXT_TAG)
        end_idx = raw_text.find(_ENDOF_TEXT_TAG)
        translated_text = raw_text[start_idx:end_idx].strip()
        if translated_text == "":
            # Same language detected, return original
            log.info(
                json.dumps(
                    {
                        "event": "translation_not_needed",
                    }
                )
            )
            return text_to_translate
        log.info(
            json.dumps(
                {
                    "event": "translation_success",
                }
            )
        )
        return translated_text

    log.warning(
        json.dumps(
            {
                "event": "translation_warning",
                "reason": "Response does not contain expected tags",
                "raw_response": raw_text,
            }
        )
    )
    return text_to_translate
