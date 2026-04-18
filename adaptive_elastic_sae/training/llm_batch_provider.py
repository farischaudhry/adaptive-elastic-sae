from __future__ import annotations

import torch

from adaptive_elastic_sae.data.llm_streamer import PythiaActivationStreamer
from adaptive_elastic_sae.training.trainer_utils import BatchProvider


class LLMActivationBatchProvider(BatchProvider):
    """BatchProvider adapter backed by a streaming LLM activation source.
    
    Uses a circular buffer to avoid GPU memory fragmentation from repeated tensor
    concatenation. Buffer is pre-allocated once and reused.
    """

    def __init__(self, streamer: PythiaActivationStreamer, buffer_size: int = 200000) -> None:
        self.streamer = streamer
        self.buffer_size = buffer_size
        self._buffer: torch.Tensor | None = None
        self._write_pos: int = 0
        self._valid_items: int = 0
        self.tokens_seen: int = 0

    def _ensure_buffer(
        self,
        d_model: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        """Lazily allocate circular buffer once on first activation block."""
        if self._buffer is None:
            self._buffer = torch.zeros(
                (self.buffer_size, d_model),
                device=device,
                dtype=dtype,
            )

    def _append_block(self, block: torch.Tensor) -> None:
        """Add activation block to circular buffer without copying existing data."""
        if self._buffer is None:
            self._ensure_buffer(block.shape[-1], block.device, block.dtype)

        write_block = block.to(device=self._buffer.device, dtype=self._buffer.dtype)

        # If a block is larger than capacity, keep only the most recent rows.
        if write_block.shape[0] > self.buffer_size:
            write_block = write_block[-self.buffer_size :]
        
        n_new = write_block.shape[0]
        space_left = self.buffer_size - self._write_pos

        if n_new <= space_left:
            # Fits contiguously without wrap.
            self._buffer[self._write_pos : self._write_pos + n_new] = write_block
        else:
            # Wrap around and split into tail/head writes.
            self._buffer[self._write_pos :] = write_block[:space_left]
            remaining = n_new - space_left
            self._buffer[:remaining] = write_block[space_left:]

        self._write_pos = (self._write_pos + n_new) % self.buffer_size
        self._valid_items = min(self._valid_items + n_new, self.buffer_size)

    def _buffer_size(self) -> int:
        return self._valid_items

    def next_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        # Fill buffer to have at least batch_size items
        while self._buffer_size() < batch_size:
            self._append_block(self.streamer.next_activation_block())

        assert self._buffer is not None
        
        # Compute read position (oldest valid item in circular buffer)
        read_pos = (self._write_pos - self._valid_items) % self.buffer_size
        
        # Extract batch, handling circular wraparound
        if read_pos + batch_size <= self.buffer_size:
            # No wraparound: contiguous slice
            x = self._buffer[read_pos : read_pos + batch_size].clone()
        else:
            # Wraparound: concatenate two slices
            first_part = self._buffer[read_pos :].clone()
            second_part = self._buffer[: batch_size - first_part.shape[0]].clone()
            x = torch.cat([first_part, second_part], dim=0)
        
        # Advance read position
        self._valid_items -= batch_size
        self.tokens_seen += int(batch_size)
        return {"x": x.to(device=device, dtype=dtype)}
