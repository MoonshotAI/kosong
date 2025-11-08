import json
import os
import random
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

from kosong.chat_provider import ChatProvider
from kosong.chat_provider.kimi import Kimi

if TYPE_CHECKING:

    def type_check(chaos: "ChaosChatProvider"):
        _: ChatProvider = chaos


class ChaosConfig(BaseModel):
    """Configuration for chaos provider."""

    error_probability: float = 0.3
    error_types: list[int] = [429, 500, 502, 503]
    retry_after: int = 2
    seed: int | None = None

    @classmethod
    def from_env(cls) -> "ChaosConfig":
        """Create config from environment variables."""
        seed_str = os.getenv("CHAOS_SEED")
        return cls(
            error_probability=float(os.getenv("CHAOS_ERROR_PROBABILITY", "0.3")),
            error_types=[
                int(x.strip()) for x in os.getenv("CHAOS_ERROR_TYPES", "429,500,502,503").split(",")
            ],
            retry_after=int(os.getenv("CHAOS_RETRY_AFTER", "2")),
            seed=int(seed_str) if seed_str else None,
        )


class ChaosTransport(httpx.AsyncBaseTransport):
    """HTTP transport that randomly injects errors."""

    def __init__(self, wrapped_transport: httpx.AsyncBaseTransport, config: ChaosConfig):
        self._wrapped = wrapped_transport
        self._config = config
        if config.seed is not None:
            random.seed(config.seed)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._should_inject_error():
            error_code = random.choice(self._config.error_types)
            return self._create_error_response(request, error_code)

        return await self._wrapped.handle_async_request(request)

    def _should_inject_error(self) -> bool:
        return random.random() < self._config.error_probability

    def _create_error_response(self, request: httpx.Request, status_code: int) -> httpx.Response:
        error_messages = {
            429: {"error": {"code": "rate_limit_exceeded", "message": "Rate limit exceeded"}},
            500: {"error": {"code": "internal_error", "message": "Internal server error"}},
            502: {"error": {"code": "bad_gateway", "message": "Bad gateway"}},
            503: {
                "error": {
                    "code": "service_unavailable",
                    "message": "Service temporarily unavailable",
                }
            },
        }

        content = json.dumps(
            error_messages.get(status_code, {"error": {"message": "Unknown error"}})
        )
        headers = {"content-type": "application/json"}

        if status_code == 429:
            headers["retry-after"] = str(self._config.retry_after)

        return httpx.Response(
            status_code=status_code,
            headers=headers,
            content=content.encode(),
            request=request,
        )


class ChaosChatProvider(Kimi):
    """Kimi chat provider with chaos error injection."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        chaos_config: ChaosConfig | None = None,
        **client_kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **client_kwargs)
        self._chaos_config = chaos_config or ChaosConfig.from_env()
        self._monkey_patch_client()

    def _monkey_patch_client(self):
        """Inject chaos transport into the client."""
        original_transport = self.client._client._transport  # pyright: ignore[reportPrivateUsage]
        chaos_transport = ChaosTransport(original_transport, self._chaos_config)
        self.client._client._transport = chaos_transport  # pyright: ignore[reportPrivateUsage]

    @property
    def model_name(self) -> str:
        if self._chaos_config.error_probability > 0:
            return f"chaos({super().model_name})"
        return super().model_name
