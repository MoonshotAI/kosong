from typing import Literal, override

from pydantic import BaseModel

from kosong.base.message import ContentPart


class ThinkPart(ContentPart):
    """
    >>> ThinkPart(think="I think I need to think about this.").model_dump()
    {'type': 'think', 'think': 'I think I need to think about this.'}
    """

    type: Literal["think"] = "think"
    think: str

    @override
    def merge_in_place(self, other) -> bool:
        if not isinstance(other, ThinkPart):
            return False
        self.think += other.think
        return True


class ImageURLPart(ContentPart):
    """
    >>> ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")).model_dump()
    {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png', 'id': None}}
    """

    class ImageURL(BaseModel):
        url: str
        """The URL of the image, can be data URI scheme like `data:image/png;base64,...`."""
        id: str | None = None
        """The ID of the image, to allow LLMs to distinguish different images."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class AudioURLPart(ContentPart):
    """
    >>> AudioURLPart(audio_url=AudioURLPart.AudioURL(url="https://example.com/audio.mp3")).model_dump()
    {'type': 'audio_url', 'audio_url': {'url': 'https://example.com/audio.mp3', 'id': None}}
    """

    class AudioURL(BaseModel):
        url: str
        """The URL of the audio, can be data URI scheme like `data:audio/aac;base64,...`."""
        id: str | None = None
        """The ID of the audio, to allow LLMs to distinguish different audios."""

    type: Literal["audio_url"] = "audio_url"
    audio_url: AudioURL
