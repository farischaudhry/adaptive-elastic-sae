from __future__ import annotations

import torch

from adaptive_elastic_sae.data.llm_streamer import PythiaActivationStreamer
from adaptive_elastic_sae.training.trainer_utils import BatchProvider


class LLMActivationBatchProvider(BatchProvider):
    """BatchProvider adapter backed by a streaming LLM activation source."""

    def __init__(self, streamer: PythiaActivationStreamer) -> None:
        self.streamer = streamer
        self._buffer: torch.Tensor | None = None
        self.tokens_seen: int = 0

    def _buffer_size(self) -> int:
        if self._buffer is None:
            return 0
        return int(self._buffer.shape[0])

    def _append_block(self, block: torch.Tensor) -> None:
        if self._buffer is None:
            self._buffer = block
        else:
            self._buffer = torch.cat([self._buffer, block], dim=0)

    def next_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        while self._buffer_size() < batch_size:
            self._append_block(self.streamer.next_activation_block())

        assert self._buffer is not None
        x = self._buffer[:batch_size]
        self._buffer = self._buffer[batch_size:]

        self.tokens_seen += int(batch_size)
        return {"x": x.to(device=device, dtype=dtype)}
