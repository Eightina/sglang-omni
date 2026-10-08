# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o's checkpoint-native streaming TTS scheduler.

The normal SGLang talker consumes the complete thinker sequence as one prompt.
This scheduler instead uses the checkpoint's ``TTSStreamingGenerator``: it
accepts thinker token/hidden pairs in small text chunks and emits codec tokens
while the thinker is still producing later text. It is intentionally a separate
stage so the normal talker remains the default fallback.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import TALKER_STAGE
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as emit_profile_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)

_REMOTE_MODULE_NAME = "sglang_omni_minicpmo_streaming_remote"


@dataclass
class StreamingTalkerState:
    """One request's buffered thinker conditions and generated codec tokens."""

    payload: StagePayload
    token_ids: list[int] = field(default_factory=list)
    hidden_rows: list[torch.Tensor] = field(default_factory=list)
    processed_rows: int = 0
    codec_chunks: list[torch.Tensor] = field(default_factory=list)
    generator: Any | None = None


def load_remote_tts_module(model_path: str) -> ModuleType:
    """Load the checkpoint's Python implementation under a stable package name."""
    cached = sys.modules.get(_REMOTE_MODULE_NAME)
    if isinstance(cached, ModuleType):
        return cached
    root = Path(model_path)
    source = root / "modeling_minicpmo.py"
    if not source.is_file():
        raise FileNotFoundError(f"MiniCPM-o streaming source is missing: {source}")
    spec = importlib.util.spec_from_file_location(
        _REMOTE_MODULE_NAME,
        source,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MiniCPM-o streaming source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_REMOTE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_streaming_tts_model(
    model_path: str, *, device: torch.device, dtype: torch.dtype
) -> tuple[ModuleType, torch.nn.Module]:
    """Construct checkpoint-native MiniCPMTTS and load only ``tts.`` weights."""
    remote = load_remote_tts_module(model_path)
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as fp:
        tts_config_data = json.load(fp).get("tts_config")
    if not isinstance(tts_config_data, dict):
        raise ValueError("MiniCPM-o config.json does not contain a tts_config mapping")
    config = remote.MiniCPMTTSConfig(**tts_config_data)
    # The published configuration leaves these sampling defaults in the caller.
    config.top_p = 0.85
    config.top_k = 25
    config.repetition_penalty = 1.05
    model = remote.MiniCPMTTS(config, audio_tokenizer=None)
    # Current transformers LlamaConfig may omit this legacy attribute. The
    # checkpoint uses full attention, but the streaming generator reads it when
    # setting up its optional reindex path.
    model.model.config.rope_theta = getattr(model.model.config, "rope_theta", 10000.0)
    weights = load_weights_by_prefix(model_path, prefix="tts.")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "MiniCPM-o streaming TTS weight mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return remote, model.to(device=device, dtype=dtype).eval()


class MiniCPMStreamingTalkerScheduler(StreamingSimpleScheduler):
    """Feed checkpoint-native streaming TTS from thinker token/hidden chunks."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        dtype: str | torch.dtype | None = None,
        text_chunk_tokens: int = 8,
        start_min_tokens: int = 16,
        max_audio_tokens_per_text_chunk: int = 500,
        enabled: bool = False,
    ) -> None:
        super().__init__(None, abort_callback=self.abort_request)
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = resolve_dtype(dtype) or torch.bfloat16
        self.text_chunk_tokens = max(int(text_chunk_tokens), 1)
        self.start_min_tokens = max(int(start_min_tokens), self.text_chunk_tokens)
        self.max_audio_tokens_per_text_chunk = max(
            int(max_audio_tokens_per_text_chunk), 1
        )
        self.enabled = bool(enabled)
        self.states: dict[str, StreamingTalkerState] = {}
        self.remote: ModuleType | None = None
        self.model: torch.nn.Module | None = None

    def ensure_model(self) -> tuple[ModuleType, torch.nn.Module]:
        if self.remote is None or self.model is None:
            logger.info(
                "Loading checkpoint-native MiniCP-o streaming TTS on %s", self.device
            )
            self.remote, self.model = load_streaming_tts_model(
                self.model_path, device=self.device, dtype=self.dtype
            )
            logger.info("MiniCPM-o streaming TTS is ready on %s", self.device)
        return self.remote, self.model

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        if not self.enabled:
            raise RuntimeError(
                "MiniCP-o early speech is disabled; start the service with "
                "--streaming_talker.factory.enable_talker_partial_start true"
            )
        self.states[request_id] = StreamingTalkerState(payload=payload)

    def validate_stream_chunk_item(self, request_id: str, item: object) -> StreamItem:
        item = super().validate_stream_chunk_item(request_id, item)
        if not isinstance(item.data, torch.Tensor):
            raise TypeError(
                "MiniCPM-o streaming talker expects a hidden-state tensor, got "
                f"{type(item.data).__name__} for {request_id!r}"
            )
        if not isinstance(item.metadata, dict) or "token_id" not in item.metadata:
            raise ValueError(
                "MiniCPM-o streaming talker requires stream metadata token_id"
            )
        return item

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self.states.get(request_id)
        if state is None:
            raise RuntimeError(
                f"MiniCPM-o streaming talker received a chunk before payload for "
                f"{request_id!r}"
            )
        token_id = int(item.metadata["token_id"])
        hidden = item.data.detach().reshape(-1, item.data.shape[-1])
        if hidden.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o streaming talker expects one hidden row per token, got "
                f"{hidden.shape[0]}"
            )
        state.token_ids.append(token_id)
        state.hidden_rows.append(hidden)
        # Keep one full chunk buffered. That look-ahead proves an earlier chunk
        # is not the terminal text chunk, so text EOS is appended only once.
        while (
            len(state.token_ids) - state.processed_rows
            >= self.start_min_tokens
        ):
            self.process_chunk(state, self.text_chunk_tokens, text_finished=False)
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.states.get(request_id)
        if state is None:
            raise RuntimeError(
                f"MiniCPM-o streaming talker finished without payload for {request_id!r}"
            )
        remaining = len(state.token_ids) - state.processed_rows
        if remaining <= 0:
            raise RuntimeError(
                "MiniCPM-o streaming talker finished without any unprocessed text"
            )
        # Process intermediate full chunks first; the final call alone receives
        # text_finished=True and therefore owns the text-EOS boundary.
        while remaining > self.text_chunk_tokens:
            self.process_chunk(state, self.text_chunk_tokens, text_finished=False)
            remaining = len(state.token_ids) - state.processed_rows
        self.process_chunk(state, remaining, text_finished=True)
        codec = (
            torch.cat(state.codec_chunks).to("cpu")
            if state.codec_chunks
            else torch.empty(0, dtype=torch.long)
        )
        result_state = MiniCPMOPipelineState(
            engine_outputs={TALKER_STAGE: {"codec_tokens": codec}}
        )
        payload = StagePayload(
            request_id=state.payload.request_id,
            request=state.payload.request,
            data=result_state.to_dict(),
        )
        emit_profile_event(
            request_id=request_id,
            stage="streaming_talker",
            event_name="streaming_talker_finished",
            metadata={"codec_token_count": int(codec.numel())},
        )
        return [OutgoingMessage(request_id=request_id, type="result", data=payload)]

    def process_chunk(
        self,
        state: StreamingTalkerState,
        count: int,
        *,
        text_finished: bool,
    ) -> None:
        if count <= 0:
            return
        remote, model = self.ensure_model()
        start = state.processed_rows
        end = start + count
        token_ids = torch.tensor(
            state.token_ids[start:end], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        hidden = torch.cat(state.hidden_rows[start:end], dim=0).to(
            device=self.device, dtype=model.emb_text.weight.dtype
        ).unsqueeze(0)
        projected = model.projector_semantic(hidden)
        if bool(model.config.normalize_projected_hidden):
            projected = F.normalize(projected, p=2, dim=-1)
        condition = model.emb_text(token_ids) + projected
        emit_profile_event(
            request_id=state.payload.request_id,
            stage="streaming_talker",
            event_name="streaming_talker_text_chunk_started",
            metadata={"text_token_count": count, "text_finished": text_finished},
        )
        if state.generator is None:
            logits_warpers, logits_processors = remote.gen_logits(
                num_code=model.num_audio_tokens,
                top_p=float(model.config.top_p),
                top_k=int(model.config.top_k),
                repetition_penalty=float(model.config.repetition_penalty),
            )
            state.generator = remote.TTSStreamingGenerator(
                model=model,
                temperature=0.8,
                eos_token=model.num_audio_tokens - 1,
                chunk_size=int(model.config.streaming_audio_chunk_size),
                logits_processors=logits_processors,
                logits_warpers=logits_warpers,
            )
        for codes, _ in state.generator.generate_with_buffer(
            condition=condition,
            text_finished=text_finished,
            max_new_token=self.max_audio_tokens_per_text_chunk,
        ):
            if codes.numel() > 0:
                state.codec_chunks.append(codes.reshape(-1).detach())
        state.processed_rows = end
        emit_profile_event(
            request_id=state.payload.request_id,
            stage="streaming_talker",
            event_name="streaming_talker_text_chunk_processed",
            metadata={
                "text_token_count": count,
                "text_finished": text_finished,
                "codec_token_count": sum(
                    int(chunk.numel()) for chunk in state.codec_chunks
                ),
            },
        )

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    def abort_request(self, request_id: str) -> None:
        self.states.pop(request_id, None)


def create_streaming_talker_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
    text_chunk_tokens: int = 8,
    start_min_tokens: int = 16,
    max_audio_tokens_per_text_chunk: int = 500,
    enable_talker_partial_start: bool = False,
) -> MiniCPMStreamingTalkerScheduler:
    """Build the lazy-loading checkpoint-native streaming Talker stage."""
    if device is None:
        device = f"cuda:{0 if gpu_id is None else gpu_id}"
    return MiniCPMStreamingTalkerScheduler(
        model_path,
        device=device,
        dtype=dtype,
        text_chunk_tokens=text_chunk_tokens,
        start_min_tokens=start_min_tokens,
        max_audio_tokens_per_text_chunk=max_audio_tokens_per_text_chunk,
        enabled=enable_talker_partial_start,
    )
