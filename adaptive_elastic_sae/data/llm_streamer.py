from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch


@dataclass
class LLMStreamConfig:
    """Configuration for streaming text and extracting LLM activations."""

    tl_model_name: str = "pythia-70m-deduped"
    hf_tokenizer_name: str = "EleutherAI/pythia-70m-deduped"
    dataset_name: str = "NeelNanda/pile-10k"
    dataset_split: str = "train"
    text_field: str = "text"
    hook_layer: int = 3
    hook_name: str | None = None
    seq_len: int = 128
    lm_batch_size: int = 8
    streaming: bool = True
    skip_docs: int = 0
    take_docs: int | None = None
    loop_dataset: bool = True
    model_dtype: str = "float32"
    device: str | torch.device = "cuda"


class PythiaActivationStreamer:
    """
    Streams text from Hugging Face datasets and extracts TransformerLens activations.
    Uses continuous token packing for 0% pad tokens and stable context windows.
    """

    def __init__(
        self,
        cfg: LLMStreamConfig,
        shared_model: Any | None = None,
        shared_tokenizer: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.hook_name = cfg.hook_name or f"blocks.{cfg.hook_layer}.hook_resid_post"

        self._tokenizer = shared_tokenizer if shared_tokenizer is not None else self._load_tokenizer()
        self._model = shared_model if shared_model is not None else self._load_model()
        self._dataset_iter = self._load_dataset_iterator()

        # Buffer State
        self._token_buffer: list[torch.Tensor] = []
        self._buffered_token_count: int = 0

        # Safe EOS Resolution
        eos_id = self._tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self._tokenizer.sep_token_id
        if eos_id is None:
            raise ValueError(
                f"Tokenizer {cfg.hf_tokenizer_name} lacks eos_token_id and sep_token_id."
            )
        self.eos_token_id = eos_id

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.hf_tokenizer_name,
            use_fast=True,
        )
        return tokenizer

    def _load_model(self):
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(
            self.cfg.tl_model_name,
            device=str(self.device),
            dtype=self.cfg.model_dtype,
        )
        model.eval()
        return model

    def _load_dataset_iterator(self) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(
            self.cfg.dataset_name,
            split=self.cfg.dataset_split,
            streaming=self.cfg.streaming,
        )

        if self.cfg.skip_docs > 0:
            ds = ds.skip(self.cfg.skip_docs)

        if self.cfg.take_docs is not None:
            ds = ds.take(self.cfg.take_docs)

        return iter(ds)

    def _next_token_batch(self) -> torch.Tensor:
        """Pulls documents, packs them continuously, and slices an exact batch."""
        target_total_tokens = self.cfg.lm_batch_size * self.cfg.seq_len

        # Fill buffer tracking count
        while self._buffered_token_count < target_total_tokens:
            try:
                sample = next(self._dataset_iter)
            except StopIteration:
                if not self.cfg.loop_dataset:
                    raise StopIteration("End of dataset reached (loop_dataset=False)")
                self._dataset_iter = self._load_dataset_iterator()
                continue

            text = sample.get(self.cfg.text_field, "")
            if not text:
                continue

            # Tokenize cleanly without auto-added boundaries
            tokens = self._tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].squeeze(0)

            # Explicitly append single boundary
            eos = torch.tensor([self.eos_token_id], dtype=tokens.dtype)
            packed_chunk = torch.cat([tokens, eos])

            self._token_buffer.append(packed_chunk)
            self._buffered_token_count += packed_chunk.numel()

        # Flatten and Slice
        flat_buffer = torch.cat(self._token_buffer)
        batch_tokens = flat_buffer[:target_total_tokens]
        remainder = flat_buffer[target_total_tokens:]

        # Safe Remainder Update
        if remainder.numel() > 0:
            self._token_buffer = [remainder]
            self._buffered_token_count = remainder.numel()
        else:
            self._token_buffer = []
            self._buffered_token_count = 0

        return batch_tokens.view(self.cfg.lm_batch_size, self.cfg.seq_len).to(
            self.device
        )

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def next_token_batch(self) -> torch.Tensor:
        return self._next_token_batch()

    @torch.no_grad()
    def next_activation_block(self) -> torch.Tensor:
        tokens = self._next_token_batch()
        captured: torch.Tensor | None = None

        def _capture_hook(act: torch.Tensor, _hook: Any) -> None:
            nonlocal captured
            captured = act

        self._model.run_with_hooks(
            tokens,
            return_type=None,  # Skip vocabulary logit calculation
            fwd_hooks=[(self.hook_name, _capture_hook)],
        )

        if captured is None:
            raise RuntimeError(f"Failed to capture hook activations for {self.hook_name}")

        return captured.reshape(-1, captured.shape[-1]).detach()
