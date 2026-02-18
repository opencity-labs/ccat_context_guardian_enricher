from typing import Any, Dict, List, Optional
import re
import uuid
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, ParseResult

# Default translations for the standard fallback message
DEFAULT_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry, I can't help with this request. Try writing short, complete questions with one request at a time.",
    "es": "Lo siento, no puedo ayudarte con esta solicitud. Intenta escribir preguntas cortas y completas con una sola solicitud a la vez.",
    "fr": "Désolé, je ne peux pas vous aider pour cette demande. Essayez d'écrire des questions courtes et complètes, une demande à la fois.",
    "de": "Es tut mir leid, ich kann bei dieser Anfrage nicht helfen. Versuchen Sie, kurze, vollständige Fragen zu stellen, jeweils eine Anfrage.",
    "it": "Mi spiace, non riesco ad aiutarti per questa richiesta. Prova a scrivere domande complete e brevi con una richiesta alla volta.",
    "pt": "Desculpe, não consigo ajudar com este pedido. Tente escrever perguntas curtas e completas, com um pedido de cada vez.",
    "zh": "抱歉，我无法处理此请求。请尝试以一次一项请求的方式，撰写简短且完整的问题。",
    "ja": "申し訳ありませんが、このリクエストにはお手伝いできません。短く完結した質問を、ひとつずつ記入してください。",
    "ko": "죄송합니다. 이 요청에는 도움을 드릴 수 없습니다. 짧고 완전한 질문을 한 번에 하나씩 작성해 보세요.",
    "ru": "Извините, я не могу помочь с этим запросом. Попробуйте задавать короткие и полные вопросы по одному.",
    "ar": "عذرًا، لا أستطيع المساعدة в هذا الطلب. حاول كتابة أسئلة قصيرة ومكتملة، واطرح طلبًا واحدًا في كل مرة.",
    "hi": "माफ़ कीजिये, मैं इस अनुरोध में मदद नहीं कर सकता। कोशिश करें कि संक्षिप्त और पूर्ण प्रश्न लिखें, एक बार में केवल एक अनुरोध।",
}


def add_utm_tracking_to_url(url: str, utm_source: str) -> str:
    """
    Add UTM tracking parameter to a URL if it doesn't already have utm_source.

    Args:
        url: The URL to add UTM tracking to
        utm_source: The UTM source parameter value

    Returns:
        The URL with UTM tracking added (if applicable)
    """

    # Remove (offset)/X pattern from url
    url = re.sub(r"/\(offset\)/\d+", "", url)

    if not utm_source:  # Skip UTM tracking if utm_source is empty
        return url

    parsed: ParseResult = urlparse(url)
    query_params: Dict[str, List[str]] = parse_qs(parsed.query)

    if "utm_source" not in query_params:
        query_params["utm_source"] = [utm_source]
        new_query: str = urlencode(query_params, doseq=True)
        parsed = parsed._replace(query=new_query)
        return urlunparse(parsed)

    return url


def enrich_links_with_utm(text: str, utm_source: str = "") -> str:
    """
    Find all HTTP/HTTPS URLs in text and add UTM tracking.
    Works with both markdown links and plain URLs.

    Args:
        text: The text containing URLs to be enriched
        utm_source: The UTM source parameter value

    Returns:
        The text with URLs enriched with UTM tracking
    """
    if not utm_source:  # Skip UTM tracking if utm_source is empty
        return text

    # Step 1: Find and temporarily replace all existing markdown links
    markdown_links = {}

    def store_markdown_link(match: re.Match[str]) -> str:
        link_text = match.group(1)
        url = match.group(2)
        title_part = match.group(3) or ""
        enhanced_url = add_utm_tracking_to_url(url, utm_source)
        placeholder = f"__MARKDOWN_LINK_{uuid.uuid4().hex}__"
        markdown_links[placeholder] = f"[{link_text}]({enhanced_url}{title_part})"
        return placeholder

    # Match markdown links with proper URL handling including parentheses, optional whitespace and titles
    markdown_link_pattern = r'\[([^\]]*)\]\(\s*(https?://[^\s)]*(?:\([^)]*\)[^\s)]*)*[^\s)]*)(\s*(?:["\'].*?["\'])?\s*)\)'
    text = re.sub(markdown_link_pattern, store_markdown_link, text)

    # Step 2: Process plain URLs with smart naming
    def convert_plain_url_to_markdown(match: re.Match[str]) -> str:
        url = match.group(0)
        suffix = ""

        # Clean trailing punctuation
        while url and url[-1] in ".,!?;":
            suffix = url[-1] + suffix
            url = url[:-1]

        # Handle parentheses: if the URL ends with ')' and has unbalanced parentheses,
        # assume the last ')' is part of the surrounding text, not the URL.
        while url.endswith(")") and url.count("(") < url.count(")"):
            suffix = url[-1] + suffix
            url = url[:-1]

        # Add UTM tracking to the URL
        enhanced_url = add_utm_tracking_to_url(url, utm_source)

        # Generate name from URL using the new logic
        # Remove query parameters and fragments for name generation
        base_url = url.split("?")[0].split("#")[0]
        url_parts = base_url.rstrip("/").split("/")

        # Get the last part of the URL path for the name (the page/section name)
        if len(url_parts) > 3 and url_parts[-1]:  # Has a meaningful path after domain
            name = url_parts[-1]
        elif (
            len(url_parts) > 4 and url_parts[-2]
        ):  # Fallback to second-to-last if last is empty
            name = url_parts[-2]
        else:
            # Fallback to domain name if path is too short
            name = url_parts[2] if len(url_parts) > 2 else url

        # Clean up the name: replace underscores and hyphens with spaces
        name = name.replace("_", " ").replace("-", " ")

        # Capitalize first letter of each word
        name = " ".join(word.capitalize() for word in name.split())

        # If name is empty or too short, use the full URL
        if not name or len(name) < 3:
            name = enhanced_url

        return f"[{name}]({enhanced_url}){suffix}"

    # Match plain URLs that are not placeholders
    plain_url_pattern = r"https?://[^\s]*(?:\([^)]*\)[^\s]*)*[^\s]*"
    text = re.sub(plain_url_pattern, convert_plain_url_to_markdown, text)

    # Step 3: Restore the processed markdown links
    for placeholder, link in markdown_links.items():
        text = text.replace(placeholder, link)

    return text


def _get_browser_lang_code_from_info(info: Any) -> str:
    """Extract the two-letter browser language code from `info` if present."""
    if isinstance(info, dict) and info.get("browser_lang"):
        return info.get("browser_lang").split("-")[0].lower()
    return ""


def select_default_message(settings: Dict[str, Any], info: Any) -> str:
    """Select the localized default message.

    Priority:
      1. settings['default_messages'][<lang>] if provided and contains language
      2. DEFAULT_MESSAGES[<lang>] if available
      3. settings['default_message'] if provided
      4. English fallback
    """
    # obtain messages dict from settings if provided
    messages = settings.get("default_messages")
    fallback_flat = settings.get("default_message")

    # determine lang code from info
    lang = _get_browser_lang_code_from_info(info)

    # Try settings-provided dict first
    if isinstance(messages, dict) and lang and lang in messages:
        return messages.get(lang)

    # Try built-in defaults
    if lang and lang in DEFAULT_MESSAGES:
        return DEFAULT_MESSAGES[lang]

    # Fallbacks
    if isinstance(messages, dict) and messages:
        # pick english if present otherwise first available
        return messages.get("en") or next(iter(messages.values()))

    if fallback_flat:
        return fallback_flat

    return DEFAULT_MESSAGES["en"]
