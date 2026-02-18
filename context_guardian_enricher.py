from typing import Any, Dict, List, Set, Optional
from cat.mad_hatter.decorators import hook
from cat.looking_glass.stray_cat import StrayCat
from cat.convo.messages import CatMessage, Role
from datetime import datetime
import re

from .audio_guardian import handle_audio_transcription
from .language_guardian import is_same_language, translate_text
from .utils import (
    add_utm_tracking_to_url,
    enrich_links_with_utm,
    select_default_message,
)


@hook
def cat_recall_query(user_message: str, cat: StrayCat) -> str:
    """
    Enhance memory search query by combining current message with recent conversation history.
    This allows for better context understanding when searching for relevant memories.

    Args:
        user_message: The current user message
        cat: The StrayCat instance

    Returns:
        Enhanced query string that includes conversation history context
    """
    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    # Check if conversation history enhancement is enabled
    use_conversation_history: bool = settings.get("use_conversation_history", True)
    if not use_conversation_history:
        return user_message

    # Get number of previous messages to include in context
    history_length: int = settings.get("conversation_history_length", 3)

    # Build enhanced query with conversation history
    enhanced_query_parts: List[str] = []

    # Add recent conversation history if available
    if hasattr(cat.working_memory, "history") and cat.working_memory.history:
        # Get recent messages (excluding the current one which hasn't been added yet)
        recent_messages = (
            cat.working_memory.history[-history_length:]
            if len(cat.working_memory.history) >= history_length
            else cat.working_memory.history
        )

        # Extract text from recent messages and add to context
        for msg in recent_messages:
            # Only include textual messages authored by the user (not the cat)
            if (
                hasattr(msg, "text")
                and msg.text.strip()
                and (
                    getattr(msg, "role", None) == Role.Human
                    or getattr(msg, "who", "").lower() == "human"
                )
            ):
                # Clean up message text (remove timestamp info that was added in before_cat_reads_message)
                clean_text = msg.text
                if "\n\ncurrent time:" in clean_text:
                    clean_text = clean_text.split("\n\ncurrent time:")[0]
                enhanced_query_parts.append(clean_text.strip())

    # Add current message
    enhanced_query_parts.append(user_message)

    # Combine all parts into enhanced query
    enhanced_query: str = " ".join(enhanced_query_parts)

    # Limit query length to avoid embedding model limits
    max_query_length: int = settings.get("max_query_length", 1000)
    if len(enhanced_query) > max_query_length:
        enhanced_query = enhanced_query[-max_query_length:]

    return enhanced_query


@hook
def fast_reply(_: Dict[str, Any], cat: StrayCat) -> Optional[CatMessage]:
    """
    Early interception of user query to decide whether to proceed with RAG+LLM.
    Can act as a panic button that always returns a default message.
    Also checks for procedural memories (tools/forms) before rejecting queries.

    Args:
        _: The input from the agent (unused)
        cat: The StrayCat instance

    Returns:
        CatMessage if no relevant context found or panic button is enabled, None otherwise
    """
    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    # handle audio message
    if cat.working_memory.user_message_json.get("audio") is not None:
        if settings.get("handle_audio", "false") == "false":
            return CatMessage(user_id=cat.user_id, text="Audio input not supported.")

        transcription = handle_audio_transcription(
            cat.working_memory.user_message_json.audio, cat
        )
        if transcription:
            cat.working_memory.user_message_json.text = transcription

    # prepare localized default message using browser info if available
    info = getattr(cat.working_memory.user_message_json, "info", {})
    default_msg = select_default_message(settings, info)

    # Check if panic button is enabled - if so, return immediately with panic text
    if settings.get("panic_button_enabled", False):
        return CatMessage(
            user_id=cat.user_id,
            text=settings.get(
                "panic_button_text",
                "Sorry, I'm under maintenance right now. Please try again later.",
            ),
        )

    # return default message if the length of the user query is less than minimum length
    min_query_length: int = settings.get("min_query_length", 10)
    user_query: str = cat.working_memory.user_message_json.text.strip()
    if len(user_query) < min_query_length:
        return CatMessage(user_id=cat.user_id, text=default_msg)

    # return default message if the length of the user query exceeds maximum length (if enabled)
    max_query_len: int = settings.get("max_query_len", 500)
    if max_query_len > 0 and len(user_query) > max_query_len:
        return CatMessage(user_id=cat.user_id, text=default_msg)

    # Regular source enricher behavior
    cat.recall_relevant_memories_to_working_memory()

    # Check if we have relevant context from declarative memories
    has_declarative_context: bool = bool(cat.working_memory.declarative_memories)

    # Check if user is currently in a form session
    form_ongoing: bool = False

    if hasattr(cat.working_memory, "active_form"):
        form_ongoing = cat.working_memory.active_form is not None

    if not has_declarative_context and not form_ongoing:
        return CatMessage(user_id=cat.user_id, text=default_msg)

    return None


@hook
def before_cat_reads_message(
    user_message_json: Dict[str, Any], cat: StrayCat
) -> Dict[str, Any]:
    """
    Hook to modify user message before cat reads it.
    Appends current time to the user message.

    Args:
        user_message_json: The user message JSON object
        cat: The StrayCat instance

    Returns:
        Modified user message JSON object
    """

    # append "current time" to user message
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message_json.text += f"\n\ncurrent time: {current_time}"

    return user_message_json


@hook(priority=0)
def agent_prompt_prefix(prefix: str, cat: StrayCat) -> str:
    """
    Hook to modify the system prompt prefix.
    Replaces $BROWSER_LANG with the actual browser language.
    """
    user_message = cat.working_memory.user_message_json
    info = getattr(user_message, "info", {})

    # Default to ENGLISH if not present or empty
    browser_lang = "ENGLISH"
    if isinstance(info, dict) and info.get("browser_lang"):
        # Dictionary to map common language codes to full language names
        lang_map = {
            "en": "ENGLISH",
            "es": "SPANISH",
            "fr": "FRENCH",
            "de": "GERMAN",
            "it": "ITALIAN",
            "pt": "PORTUGUESE",
            "zh": "CHINESE",
            "ja": "JAPANESE",
            "ko": "KOREAN",
            "ru": "RUSSIAN",
            "ar": "ARABIC",
            "hi": "HINDI",
        }

        # Normalize stuff like en-US to en
        lang_code = info.get("browser_lang").split("-")[0].lower()
        browser_lang = lang_map.get(lang_code, browser_lang.upper())

    return prefix.replace("$BROWSER_LANG", browser_lang)


@hook(priority=1)
def before_cat_sends_message(message: CatMessage, cat: StrayCat) -> CatMessage:
    """
    Enrich the outgoing message with the sources used during the main (and optional double) pass.

    Args:
        message: The CatMessage to be sent
        cat: The StrayCat instance

    Returns:
        The enriched CatMessage with sources and UTM tracking
    """
    # Check if the user's message and bot's answer are in the same language
    is_same_lang = is_same_language(
        cat.working_memory.user_message_json.text, message.text
    )

    if not is_same_lang:
        # log.info("Language mismatch detected. Translating response...")
        message.text = translate_text(
            message.text, cat.working_memory.user_message_json.text, cat
        )

    # if form_ongoing: # skip rejection if user is in a form session
    form_ongoing: bool = False
    if hasattr(cat.working_memory, "active_form"):
        form_ongoing = cat.working_memory.active_form is not None
    if form_ongoing:
        return message
    if "<no_sources>" in message.text:
        message.text = message.text.replace("<no_sources>", "")
        return message

    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    utm_source: str = settings.get("utm_source", "")

    # Collect sources from declarative memories in order (most relevant first)
    sources: List[Dict[str, str]] = []
    seen_sources: Set[str] = set()
    for mem in cat.working_memory.declarative_memories:
        doc = mem[0]  # Document is the first element in the tuple
        source: Optional[str] = doc.metadata.get("source")
        title: Optional[str] = doc.metadata.get(
            "title"
        )  # Page title is available here for web pages
        if source and source not in seen_sources:
            sources.append(
                {"url": source, "label": title or "", "length": len(doc.page_content)}
            )
            seen_sources.add(source)

    if settings.get("double_pass", False):
        # Double pass: query memory with user query + generated response
        combined_text: str = (
            cat.working_memory.user_message_json.text + " " + message.text
        )
        embedding = cat.embedder.embed_query(combined_text)
        second_memories = cat.memory.vectors.declarative.recall_memories_from_embedding(
            embedding
        )

        second_sources: List[Dict[str, str]] = []
        seen_second_sources: Set[str] = set()
        for mem in second_memories:
            doc = mem[0]
            source: Optional[str] = doc.metadata.get("source")
            title: Optional[str] = doc.metadata.get("title")
            if source and source not in seen_second_sources:
                second_sources.append(
                    {
                        "url": source,
                        "label": title or "",
                        "length": len(doc.page_content),
                    }
                )
                seen_second_sources.add(source)

        # Find intersection while preserving order from main sources
        relevant_sources: List[Dict[str, str]] = [
            s for s in sources if s["url"] in seen_second_sources
        ]
        if not relevant_sources:
            relevant_sources = sources if sources else second_sources

    else:
        # Single pass: use all sources from main pass
        relevant_sources: List[Dict[str, str]] = sources

    # Filter sources by content length
    min_source_char = settings.get("min_source_char", 100)
    relevant_sources = [
        s for s in relevant_sources if s.get("length", 0) >= min_source_char
    ]

    # Extract URLs from message text if remove_inline_links_from_sources or suggestion_first is enabled
    inline_urls: Set[str] = set()
    remove_inline = settings.get("remove_inline_links_from_sources", False)
    suggestion_first = settings.get("suggestion_first", False)

    if remove_inline or suggestion_first:
        # Find all URLs in the message text (both plain and markdown)
        # Match markdown links [text](url)
        markdown_urls = re.findall(r"\[([^\]]*)\]\((https?://[^\s)]+)\)", message.text)
        inline_urls.update(url for _, url in markdown_urls)

        # Match plain URLs
        plain_urls = re.findall(r"https?://[^\s<>\[\]()]+", message.text)
        inline_urls.update(plain_urls)

        # Clean trailing punctuation from URLs
        cleaned_inline_urls: Set[str] = set()
        for url in inline_urls:
            while url and url[-1] in ".,!?;":
                url = url[:-1]
            cleaned_inline_urls.add(url)
        inline_urls = cleaned_inline_urls

    # Filter sources and add UTM tracking
    processed_sources: List[Dict[str, str]] = []
    prioritized_sources: List[Dict[str, str]] = []

    for s in relevant_sources:
        source_url = s["url"]
        is_inline = source_url in inline_urls

        # Skip if URL appears in message text and remove_inline is enabled
        if is_inline and remove_inline:
            continue

        processed_source = {
            "url": add_utm_tracking_to_url(source_url, utm_source),
            "label": s["label"].split("/")[0],
        }

        if is_inline and suggestion_first:
            prioritized_sources.append(processed_source)
        else:
            processed_sources.append(processed_source)

    if suggestion_first:
        processed_sources = prioritized_sources + processed_sources

    # Deduplicate by label - keep only first occurrence of each unique label
    seen_labels: Set[str] = set()
    unique_sources: List[Dict[str, str]] = []
    for source in processed_sources:
        label = source["label"]
        if label and label not in seen_labels:
            seen_labels.add(label)
            unique_sources.append(source)
        elif not label:  # Keep sources without labels
            unique_sources.append(source)

    message.sources = unique_sources

    # Add UTM tracking to all links in the final message
    message.text = enrich_links_with_utm(message.text, utm_source)

    return message
