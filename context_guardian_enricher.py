from typing import Any, Dict, List, Set, Tuple, Optional, Union
from cat.mad_hatter.decorators import hook
from cat.looking_glass.stray_cat import StrayCat
from cat.convo.messages import CatMessage
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, ParseResult


def add_utm_tracking_to_url(url: str, utm_source: str) -> str:
    """
    Add UTM tracking parameter to a URL if it doesn't already have utm_source.
    
    Args:
        url: The URL to add UTM tracking to
        utm_source: The UTM source parameter value
        
    Returns:
        The URL with UTM tracking added (if applicable)
    """

    if not utm_source:  # Skip UTM tracking if utm_source is empty
        return url

    parsed: ParseResult = urlparse(url)
    query_params: Dict[str, List[str]] = parse_qs(parsed.query)

    if 'utm_source' not in query_params:
        query_params['utm_source'] = [utm_source]
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
    
    import uuid
    
    # Step 1: Find and temporarily replace all existing markdown links
    markdown_links = {}
    
    def store_markdown_link(match: re.Match[str]) -> str:
        link_text = match.group(1)
        url = match.group(2)
        enhanced_url = add_utm_tracking_to_url(url, utm_source)
        placeholder = f"__MARKDOWN_LINK_{uuid.uuid4().hex}__"
        markdown_links[placeholder] = f'[{link_text}]({enhanced_url})'
        return placeholder
    
    # Match markdown links with proper URL handling including parentheses
    markdown_link_pattern = r'\[([^\]]*)\]\((https?://[^\s)]*(?:\([^)]*\)[^\s)]*)*[^\s)]*)\)'
    text = re.sub(markdown_link_pattern, store_markdown_link, text)
    
    # Step 2: Process plain URLs with smart naming
    def convert_plain_url_to_markdown(match: re.Match[str]) -> str:
        url = match.group(0)
        suffix = ""
        
        # Clean trailing punctuation
        while url and url[-1] in '.,!?;':
            suffix = url[-1] + suffix
            url = url[:-1]
            
        # Handle parentheses: if the URL ends with ')' and has unbalanced parentheses,
        # assume the last ')' is part of the surrounding text, not the URL.
        while url.endswith(')') and url.count('(') < url.count(')'):
            suffix = url[-1] + suffix
            url = url[:-1]
        
        # Add UTM tracking to the URL
        enhanced_url = add_utm_tracking_to_url(url, utm_source)
        
        # Generate name from URL using the new logic
        # Remove query parameters and fragments for name generation
        base_url = url.split('?')[0].split('#')[0]
        url_parts = base_url.rstrip('/').split('/')
        
        # Get the last part of the URL path for the name (the page/section name)
        if len(url_parts) > 3 and url_parts[-1]:  # Has a meaningful path after domain
            name = url_parts[-1]
        elif len(url_parts) > 4 and url_parts[-2]:  # Fallback to second-to-last if last is empty
            name = url_parts[-2]
        else:
            # Fallback to domain name if path is too short
            name = url_parts[2] if len(url_parts) > 2 else url
        
        # Clean up the name: replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        # If name is empty or too short, use the full URL
        if not name or len(name) < 3:
            name = enhanced_url
        
        return f'[{name}]({enhanced_url}){suffix}'
    
    # Match plain URLs that are not placeholders
    plain_url_pattern = r'https?://[^\s]*(?:\([^)]*\)[^\s]*)*[^\s]*'
    text = re.sub(plain_url_pattern, convert_plain_url_to_markdown, text)
    
    # Step 3: Restore the processed markdown links
    for placeholder, link in markdown_links.items():
        text = text.replace(placeholder, link)
    
    return text

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
    use_conversation_history: bool = settings.get('use_conversation_history', True)
    if not use_conversation_history:
        return user_message
    
    # Get number of previous messages to include in context
    history_length: int = settings.get('conversation_history_length', 3)
    
    # Build enhanced query with conversation history
    enhanced_query_parts: List[str] = []
    
    # Add recent conversation history if available
    if hasattr(cat.working_memory, 'history') and cat.working_memory.history:
        # Get recent messages (excluding the current one which hasn't been added yet)
        recent_messages = cat.working_memory.history[-history_length:] if len(cat.working_memory.history) >= history_length else cat.working_memory.history
        
        # Extract text from recent messages and add to context
        for msg in recent_messages:
            if hasattr(msg, 'text') and msg.text.strip():
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
    max_query_length: int = settings.get('max_query_length', 1000)
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
    
    # Check if panic button is enabled - if so, return immediately with panic text
    if settings.get('panic_button_enabled', False):
        return CatMessage(user_id=cat.user_id, text=settings.get('panic_button_text', "Sorry, I'm under maintenance right now. Please try again later."))

    # return default message if the length of the user query is less than minimum length
    min_query_length: int = settings.get('min_query_length', 10)
    user_query: str = cat.working_memory.user_message_json.text.strip()
    if len(user_query) < min_query_length:
        return CatMessage(user_id=cat.user_id, text=settings.get('default_message', 'Sorry, I can\'t help you.'))
    
    # return default message if the length of the user query exceeds maximum length (if enabled)
    max_query_len: int = settings.get('max_query_len', 500)
    if max_query_len > 0 and len(user_query) > max_query_len:
        return CatMessage(user_id=cat.user_id, text=settings.get('default_message', 'Sorry, I can\'t help you.'))

    # Regular source enricher behavior
    cat.recall_relevant_memories_to_working_memory()

    # Check if we have relevant context from declarative memories
    has_declarative_context: bool = bool(cat.working_memory.declarative_memories)
    
    # Check if user is currently in a form session
    form_ongoing: bool = False
    
    if hasattr(cat.working_memory, 'active_form'):
        form_ongoing = cat.working_memory.active_form is not None

    if not has_declarative_context and not form_ongoing:
        return CatMessage(user_id=cat.user_id, text=settings.get('default_message', 'Sorry, I can\'t help you.'))

    return None

@hook
def before_cat_reads_message(user_message_json: Dict[str, Any], cat: StrayCat) -> Dict[str, Any]:
    """
    Hook to modify user message before cat reads it.
    Appends current time to the user message.
    
    Args:
        user_message_json: The user message JSON object
        _: The StrayCat instance (unused)
        
    Returns:
        Modified user message JSON object
    """
    # append "current time" to user message
    from datetime import datetime
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message_json.text += f"\n\ncurrent time: {current_time}"
    return user_message_json


@hook
def before_cat_sends_message(message: CatMessage, cat: StrayCat) -> CatMessage:
    """
    Enrich the outgoing message with the sources used during the main (and optional double) pass.
    
    Args:
        message: The CatMessage to be sent
        cat: The StrayCat instance
        
    Returns:
        The enriched CatMessage with sources and UTM tracking
    """
    
    # if form_ongoing: # skip rejection if user is in a form session
    form_ongoing: bool = False
    if hasattr(cat.working_memory, 'active_form'):
        form_ongoing = cat.working_memory.active_form is not None
    if form_ongoing:
        return message
    if "<no_sources>" in message.text:
        message.text = message.text.replace("<no_sources>", "")
        return message
    
    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    utm_source: str = settings.get('utm_source', '')

    # Collect sources from declarative memories in order (most relevant first)
    sources: List[Dict[str, str]] = []
    seen_sources: Set[str] = set()
    for mem in cat.working_memory.declarative_memories:
        doc = mem[0]  # Document is the first element in the tuple
        source: Optional[str] = doc.metadata.get('source')
        title: Optional[str] = doc.metadata.get('title')  # Page title is available here for web pages
        if source and source not in seen_sources:
            sources.append({"url": source, "label": title or "", "length": len(doc.page_content)})
            seen_sources.add(source)

    if settings.get('double_pass', False):
        # Double pass: query memory with user query + generated response
        combined_text: str = cat.working_memory.user_message_json.text + " " + message.text
        embedding = cat.embedder.embed_query(combined_text)
        second_memories = cat.memory.vectors.declarative.recall_memories_from_embedding(embedding)

        second_sources: List[Dict[str, str]] = []
        seen_second_sources: Set[str] = set()
        for mem in second_memories:
            doc = mem[0]
            source: Optional[str] = doc.metadata.get('source')
            title: Optional[str] = doc.metadata.get('title')
            if source and source not in seen_second_sources:
                second_sources.append({"url": source, "label": title or "", "length": len(doc.page_content)})
                seen_second_sources.add(source)

        # Find intersection while preserving order from main sources
        relevant_sources: List[Dict[str, str]] = [s for s in sources if s['url'] in seen_second_sources]
        if not relevant_sources:
            relevant_sources = sources if sources else second_sources

    else:
        # Single pass: use all sources from main pass
        relevant_sources: List[Dict[str, str]] = sources

    # Filter sources by content length
    min_source_char = settings.get('min_source_char', 100)
    relevant_sources = [s for s in relevant_sources if s.get('length', 0) >= min_source_char]

    # Extract URLs from message text if remove_inline_links_from_sources or suggestion_first is enabled
    inline_urls: Set[str] = set()
    remove_inline = settings.get('remove_inline_links_from_sources', False)
    suggestion_first = settings.get('suggestion_first', False)

    if remove_inline or suggestion_first:
        # Find all URLs in the message text (both plain and markdown)
        # Match markdown links [text](url)
        markdown_urls = re.findall(r'\[([^\]]*)\]\((https?://[^\s)]+)\)', message.text)
        inline_urls.update(url for _, url in markdown_urls)
        
        # Match plain URLs
        plain_urls = re.findall(r'https?://[^\s<>\[\]()]+', message.text)
        inline_urls.update(plain_urls)
        
        # Clean trailing punctuation from URLs
        cleaned_inline_urls: Set[str] = set()
        for url in inline_urls:
            while url and url[-1] in '.,!?;':
                url = url[:-1]
            cleaned_inline_urls.add(url)
        inline_urls = cleaned_inline_urls

    # Filter sources and add UTM tracking
    processed_sources: List[Dict[str, str]] = []
    prioritized_sources: List[Dict[str, str]] = []

    for s in relevant_sources:
        source_url = s['url']
        is_inline = source_url in inline_urls

        # Skip if URL appears in message text and remove_inline is enabled
        if is_inline and remove_inline:
            continue
        
        processed_source = {
            "url": add_utm_tracking_to_url(source_url, utm_source), 
            "label": s['label'].split("/")[0]
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
        label = source['label']
        if label and label not in seen_labels:
            seen_labels.add(label)
            unique_sources.append(source)
        elif not label:  # Keep sources without labels
            unique_sources.append(source)
    
    message.sources = unique_sources

    # Add UTM tracking to all links in the final message
    message.text = enrich_links_with_utm(message.text, utm_source)
    return message