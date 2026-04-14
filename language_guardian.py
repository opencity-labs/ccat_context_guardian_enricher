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
    # log.warning(
    #     f"Detected languages - Text 1: {lang1['language']} (confidence: {lang1['score']}), Text 2: {lang2['language']} (confidence: {lang2['score']})"
    # )
    return lang1["language"] == lang2["language"]


_STARTOF_TEXT_TAG = "<|startoftext|>"
_ENDOF_TEXT_TAG = "<|endoftext|>"


def translate_text(text_to_translate: str, reference_text: str, cat: Any) -> str:
    """
    Translates text_to_translate to the language of reference_text
    using the LLM configured in the Cheshire Cat framework.
    """
    prompt = f"""Act as a translation engine. You will be provided with two texts, your task is to compare the language of the "Reference Text" with the "Text to Translate."
- Maintain the original tone and formatting.
- Output the result between {_STARTOF_TEXT_TAG} and {_ENDOF_TEXT_TAG}.
- No explanations, headers, or notes.

1. If both texts are in the same language: the output will be empty, so just: {_STARTOF_TEXT_TAG}{_ENDOF_TEXT_TAG};
2. If the two languages differ: the output will be ONLY the translation of the "Text to Translate" into the language used in the "Reference Text", so {_STARTOF_TEXT_TAG}TRANSLATED TEXT{_ENDOF_TEXT_TAG}

Reference Text for language: "{reference_text}"
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
