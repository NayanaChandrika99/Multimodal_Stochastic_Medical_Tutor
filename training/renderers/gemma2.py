# ABOUTME: Tinker renderer for Gemma2/MedGemma vision-language models
# ABOUTME: Converts messages to tokens for RL training with vision inputs

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

# Try to import tinker, provide fallbacks for testing
try:
    from tinker import EncodedTextChunk, ImageChunk, ModelInput

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

    # Fallback dataclasses for testing without tinker
    @dataclass
    class EncodedTextChunk:  # type: ignore[no-redef]
        """Fallback for tinker.EncodedTextChunk."""

        tokens: list[int] = field(default_factory=list)

    @dataclass
    class ImageChunk:  # type: ignore[no-redef]
        """Fallback for tinker.ImageChunk."""

        image_bytes: bytes = b""

    @dataclass
    class ModelInput:  # type: ignore[no-redef]
        """Fallback for tinker.ModelInput."""

        chunks: list[Any] = field(default_factory=list)

        def __init__(self, chunks: list[Any] | None = None) -> None:
            self.chunks = chunks or []


if TYPE_CHECKING:
    from PIL import Image
    from transformers import PreTrainedTokenizer

Message = dict[str, Any]  # Type alias for message dicts


@dataclass
class RenderedMessage:
    """Result of rendering a message to tokens."""

    header: ModelInput
    output: ModelInput
    stop_overlap: int = 0


@dataclass
class ImagePart:
    """Image content in a message."""

    image: Image.Image


@dataclass
class TextPart:
    """Text content in a message."""

    text: str


def image_to_chunk(image: Image.Image) -> ImageChunk:
    """Convert PIL Image to Tinker ImageChunk."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ImageChunk(image_bytes=buffer.getvalue())


class Gemma2VLRenderer:
    """Tinker renderer for Gemma2 vision-language models.

    Handles the Gemma2 chat template format:
    - <start_of_turn>user\\n...content...<end_of_turn>
    - <start_of_turn>model\\n...content...<end_of_turn>
    """

    # Gemma2 special tokens
    START_OF_TURN = "<start_of_turn>"
    END_OF_TURN = "<end_of_turn>"

    # Role mappings
    ROLE_MAP = {
        "user": "user",
        "assistant": "model",
        "system": "user",  # Gemma2 treats system as user with special prefix
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        max_tokens: int = 2048,
        include_system_in_user: bool = True,
    ) -> None:
        """Initialize the renderer.

        Args:
            tokenizer: Gemma2/MedGemma tokenizer
            max_tokens: Maximum tokens for generation
            include_system_in_user: Whether to prepend system message to first user message
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.include_system_in_user = include_system_in_user

        # Cache token IDs
        self._start_of_turn_id = self._get_token_id(self.START_OF_TURN)
        self._end_of_turn_id = self._get_token_id(self.END_OF_TURN)
        self._newline_id = self._get_token_id("\n")

    def _get_token_id(self, text: str) -> int:
        """Get token ID for a string, handling special tokens."""
        ids = cast(list[int], self.tokenizer.encode(text, add_special_tokens=False))
        if len(ids) == 1:
            return ids[0]
        # For multi-token strings, return first token
        return ids[0] if ids else 0

    def get_stop_sequences(self) -> list[list[int]]:
        """Return stop sequences for generation.

        Returns:
            List of token ID sequences that signal generation should stop.
        """
        # <end_of_turn> is the primary stop sequence
        end_turn_tokens = cast(
            list[int],
            self.tokenizer.encode(self.END_OF_TURN, add_special_tokens=False),
        )
        return [end_turn_tokens]

    def render_message(
        self,
        message: dict[str, Any],
        *,
        is_training: bool = False,
        system_prefix: str | None = None,
    ) -> RenderedMessage:
        """Convert a message to Tinker ModelInput.

        Args:
            message: The message to render (has role and content)
            is_training: Whether this is for training (includes output in loss)
            system_prefix: Optional system message to prepend (for user messages)

        Returns:
            RenderedMessage with header (prompt) and output (response) chunks
        """
        role = message.get("role", "user")
        content = message.get("content", "")
        gemma_role = self.ROLE_MAP.get(role, "user")

        # Build header: <start_of_turn>role\n
        header_text = f"{self.START_OF_TURN}{gemma_role}\n"
        if system_prefix and role == "user":
            header_text += f"{system_prefix}\n\n"

        header_tokens = cast(
            list[int], self.tokenizer.encode(header_text, add_special_tokens=False)
        )
        header_chunks: list[EncodedTextChunk | ImageChunk] = [
            EncodedTextChunk(tokens=header_tokens)
        ]

        # Build output: content + <end_of_turn>
        output_chunks: list[EncodedTextChunk | ImageChunk] = []

        # Process content (can be string or list of parts)
        content_parts = self._normalize_content(content)
        for part in content_parts:
            if isinstance(part, ImagePart):
                output_chunks.append(image_to_chunk(part.image))
            elif isinstance(part, TextPart):
                text_tokens = cast(
                    list[int], self.tokenizer.encode(part.text, add_special_tokens=False)
                )
                output_chunks.append(EncodedTextChunk(tokens=text_tokens))

        # Add end of turn
        end_tokens = cast(
            list[int], self.tokenizer.encode(self.END_OF_TURN, add_special_tokens=False)
        )
        output_chunks.append(EncodedTextChunk(tokens=end_tokens))

        return RenderedMessage(
            header=ModelInput(header_chunks),
            output=ModelInput(output_chunks),
            stop_overlap=len(end_tokens),
        )

    def render_messages(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> ModelInput:
        """Render a full conversation to ModelInput.

        Args:
            messages: List of messages to render
            add_generation_prompt: Whether to add assistant prompt at end

        Returns:
            ModelInput ready for generation or training
        """
        chunks: list[EncodedTextChunk | ImageChunk] = []

        # Extract system message if present
        system_prefix = None
        start_idx = 0
        if messages and messages[0].get("role") == "system":
            system_content = messages[0].get("content", "")
            if isinstance(system_content, str):
                system_prefix = system_content
            start_idx = 1

        # Render each message
        for i, message in enumerate(messages[start_idx:]):
            prefix = system_prefix if i == 0 else None
            rendered = self.render_message(message, system_prefix=prefix)
            chunks.extend(rendered.header.chunks)
            chunks.extend(rendered.output.chunks)

        # Add generation prompt for assistant response
        if add_generation_prompt:
            gen_prompt = f"{self.START_OF_TURN}model\n"
            gen_tokens = cast(
                list[int], self.tokenizer.encode(gen_prompt, add_special_tokens=False)
            )
            chunks.append(EncodedTextChunk(tokens=gen_tokens))

        return ModelInput(chunks)

    def parse_response(self, tokens: list[int]) -> str:
        """Convert generated tokens back to text.

        Args:
            tokens: Token IDs from generation

        Returns:
            Decoded text with stop sequences removed
        """
        text = cast(str, self.tokenizer.decode(tokens, skip_special_tokens=False))

        # Remove trailing end_of_turn if present
        if text.endswith(self.END_OF_TURN):
            text = text[: -len(self.END_OF_TURN)]

        return text.strip()

    def build_generation_prompt(
        self,
        messages: Sequence[Message],
    ) -> ModelInput:
        """Build prompt for generation (alias for render_messages).

        Args:
            messages: Conversation history

        Returns:
            ModelInput ready for sampling
        """
        return self.render_messages(messages, add_generation_prompt=True)

    def decode(self, tokens: list[int] | ModelInput) -> str:
        """Decode tokens or ModelInput to text.

        Args:
            tokens: Either token list or ModelInput

        Returns:
            Decoded text string
        """
        if isinstance(tokens, ModelInput):
            # Extract all tokens from chunks
            all_tokens: list[int] = []
            for chunk in tokens.chunks:
                if isinstance(chunk, EncodedTextChunk):
                    all_tokens.extend(chunk.tokens)
            return cast(str, self.tokenizer.decode(all_tokens, skip_special_tokens=False))
        return cast(str, self.tokenizer.decode(tokens, skip_special_tokens=False))

    def _normalize_content(self, content: str | list | dict) -> list[TextPart | ImagePart]:
        """Normalize message content to list of parts.

        Args:
            content: String, list of dicts, or single dict

        Returns:
            List of TextPart and ImagePart objects
        """
        if isinstance(content, str):
            return [TextPart(text=content)]

        if isinstance(content, dict):
            return self._parse_content_dict(content)

        if isinstance(content, list):
            parts: list[TextPart | ImagePart] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(TextPart(text=item))
                elif isinstance(item, dict):
                    parts.extend(self._parse_content_dict(item))
                elif isinstance(item, TextPart | ImagePart):
                    parts.append(item)
            return parts

        return [TextPart(text=str(content))]

    def _parse_content_dict(self, item: dict) -> list[TextPart | ImagePart]:
        """Parse a content dict to parts.

        Args:
            item: Dict with type/text or type/image keys

        Returns:
            List containing TextPart or ImagePart
        """
        content_type = item.get("type", "text")

        if content_type == "text":
            return [TextPart(text=str(item.get("text", "")))]

        if content_type == "image":
            image = item.get("image")
            if image is not None:
                from PIL import Image as PILImage

                if isinstance(image, PILImage.Image):
                    return [ImagePart(image=image)]

        return []
