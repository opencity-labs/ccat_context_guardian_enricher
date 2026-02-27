from enum import Enum
from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field, validator


class ContextGuardianEnricherSettings(BaseModel):
    double_pass: bool = Field(
        title="Double pass in memory",
        default=False,
        description='Whether to add all the retrieved context or doing a double pass in the memory to get "post" answer context',
    )
    utm_source: str = Field(
        title="UTM Source",
        default="",
        description="UTM source parameter to add to outgoing links for tracking. Leave empty to disable UTM tracking.",
    )
    min_query_length: int = Field(
        title="Minimum Query Length",
        default=10,
        description="Minimum number of characters a query must have to be accepted for processing",
    )
    default_message: str = Field(
        title="Default Message",
        default="Sorry, I can't help you. To answer adequately: • Write short, complete sentences • Express one request at a time",
        description="The default message to use when no declarative memory is found",
        extra={"type": "TextArea"},
    )
    panic_button_enabled: bool = Field(
        title="Panic Button Enabled",
        default=False,
        description="Enable panic button mode - always returns the panic button text regardless of context",
    )
    panic_button_text: str = Field(
        title="Panic Button Text",
        default="Sorry, I'm under maintenance right now. Please try again later.",
        description="The text to return when panic button mode is enabled",
        extra={"type": "TextArea"},
    )
    use_conversation_history: bool = Field(
        title="Use Conversation History",
        default=True,
        description="Include recent conversation history when searching memories for better context understanding",
    )
    conversation_history_length: int = Field(
        title="Conversation History Length",
        default=3,
        description="Number of previous messages to include in memory search context (0 to disable)",
    )
    max_query_length: int = Field(
        title="Maximum Query Length",
        default=1000,
        description="Maximum length of the enhanced query string to avoid embedding model limits",
    )
    max_query_len: int = Field(
        title="Maximum User Query Length",
        default=500,
        description="Maximum number of characters allowed in a user query. Set to 0 to disable this check.",
    )
    min_source_char: int = Field(
        title="Minimum Source Characters",
        default=100,
        description="Minimum number of characters a source content must have to be included in the sources list",
    )
    remove_inline_links_from_sources: bool = Field(
        title="Remove Inline Links from Sources",
        default=False,
        description="If enabled, removes from sources any URL that appears in the message text itself",
    )
    suggestion_first: bool = Field(
        title="Prioritize Inline Links in Sources",
        default=False,
        description="If enabled, moves sources that appear in the message text to the top of the sources list",
    )
    handle_audio: str = Field(
        title="Handle Audio",
        default="false",
        description="Whether to process audio attachments in the query and include their transcriptions in the context",
    )
    google_api_key: str = Field(
        title="Google API Key",
        default="",
        description="API Key for Gemini API (Audio STT)",
        extra={"type": "Password"},
    )

    class GeminiModels(str, Enum):
        GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
        GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
        GEMINI_2_5_FLASH = "gemini-2.5-flash"
        GEMINI_2_0_FLASH = "gemini-2.0-flash"

    selected_model: GeminiModels = Field(
        title="Selected Gemini Model",
        default=GeminiModels.GEMINI_2_0_FLASH,
        description="Gemini model to try first when calling the API",
    )

    @validator("handle_audio")
    def validate_handle_audio(cls, v):
        """Validate that handle_audio is a valid boolean string"""
        if v.lower() not in ("true", "false"):
            raise ValueError("handle_audio must be 'true' or 'false'")
        return v.lower()

    @validator("min_query_length")
    def validate_min_query_length(cls, v):
        """Validate that min_query_length is non-negative"""
        if not v >= 0:
            raise ValueError("Minimum query length must be non-negative")
        return v

    @validator("conversation_history_length")
    def validate_conversation_history_length(cls, v):
        """Validate that conversation_history_length is within reasonable bounds"""
        if not 0 <= v <= 10:
            raise ValueError("Conversation history length must be between 0 and 10")
        return v

    @validator("max_query_length")
    def validate_max_query_length(cls, v):
        """Validate that max_query_length is within reasonable bounds"""
        if not 100 <= v <= 5000:
            raise ValueError("Maximum query length must be between 100 and 5000")
        return v

    @validator("min_source_char")
    def validate_min_source_char(cls, v):
        """Validate that min_source_char is non-negative"""
        if not v >= 0:
            raise ValueError("Minimum source characters must be non-negative")
        return v


@plugin
def settings_model():
    return ContextGuardianEnricherSettings
