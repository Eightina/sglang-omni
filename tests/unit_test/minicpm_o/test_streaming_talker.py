# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native MiniCPM-o streaming talker scheduler contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    DECODE_STAGE,
    STREAMING_TALKER_STAGE,
    TALKER_STAGE,
    resolve_preprocessing_next_stages,
    resolve_thinker_next_stages,
    resolve_thinker_stream_done_targets,
)
from sglang_omni.models.minicpm_o.streaming_talker import (
    MiniCPMStreamingTalkerScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


class FakeGenerator:
    def __init__(self, **kwargs) -> None:
        self.calls: list[tuple[torch.Tensor, bool]] = []

    def generate_with_buffer(self, *, condition, text_finished, max_new_token):
        del max_new_token
        self.calls.append((condition.detach().clone(), text_finished))
        yield torch.tensor([[41, 42]], dtype=torch.long), text_finished


class FakeRemote:
    TTSStreamingGenerator = FakeGenerator

    @staticmethod
    def gen_logits(**kwargs):
        del kwargs
        return [], []


class FakeModel:
    def __init__(self) -> None:
        self.emb_text = nn.Embedding(1000, 4)
        self.projector_semantic = nn.Linear(6, 4, bias=False)
        self.config = SimpleNamespace(
            normalize_projected_hidden=True,
            top_p=0.85,
            top_k=25,
            repetition_penalty=1.05,
            streaming_audio_chunk_size=25,
        )
        self.num_audio_tokens = 99


def payload(request_id: str = "stream", *, streaming: bool = False) -> StagePayload:
    params = (
        {"stage_params": {STREAMING_TALKER_STAGE: {"enabled": True}}}
        if streaming
        else {}
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=None,
            params=params,
            metadata={"output_modalities": ["text", "audio"]},
        ),
        data=MiniCPMOPipelineState(
            prompt={"input_ids": torch.tensor([1, 2])}
        ).to_dict(),
    )


def item(index: int) -> StreamItem:
    return StreamItem(
        chunk_id=index,
        data=torch.full((1, 6), float(index)),
        from_stage="thinker",
        metadata={"token_id": index + 10},
    )


def scheduler(*, enabled: bool = True) -> MiniCPMStreamingTalkerScheduler:
    instance = MiniCPMStreamingTalkerScheduler(
        "unused",
        device="cpu",
        dtype=torch.float32,
        text_chunk_tokens=8,
        start_min_tokens=16,
        enabled=enabled,
    )
    model = FakeModel()
    instance.ensure_model = lambda: (FakeRemote, model)
    return instance


def test_streaming_request_routes_to_early_talker_and_skips_normal_talker() -> None:
    streaming = payload(streaming=True)
    normal = payload(streaming=False)

    assert STREAMING_TALKER_STAGE in resolve_preprocessing_next_stages(
        "stream", streaming
    )
    assert STREAMING_TALKER_STAGE not in resolve_preprocessing_next_stages(
        "normal", normal
    )
    assert resolve_thinker_next_stages("stream", streaming) == [DECODE_STAGE]
    assert resolve_thinker_next_stages("normal", normal) == [
        DECODE_STAGE,
        TALKER_STAGE,
    ]
    assert resolve_thinker_stream_done_targets("stream", streaming) == [
        DECODE_STAGE,
        STREAMING_TALKER_STAGE,
    ]


def test_streaming_talker_waits_for_lookahead_before_first_chunk() -> None:
    instance = scheduler()
    instance.on_streaming_new_request("stream", payload())

    for index in range(15):
        assert instance.on_stream_chunk("stream", item(index)) == []
    assert instance.states["stream"].processed_rows == 0

    instance.on_stream_chunk("stream", item(15))

    state = instance.states["stream"]
    assert state.processed_rows == 8
    assert state.generator is not None
    assert len(state.generator.calls) == 1
    assert state.generator.calls[0][1] is False


def test_streaming_talker_finishes_remaining_text_and_returns_codec_tokens() -> None:
    instance = scheduler()
    instance.on_streaming_new_request("stream", payload())
    for index in range(16):
        instance.on_stream_chunk("stream", item(index))

    messages = instance.on_stream_done("stream")

    assert len(messages) == 1
    result = messages[0].data
    codec = result.data["engine_outputs"]["talker"]["codec_tokens"]
    assert torch.equal(codec, torch.tensor([41, 42, 41, 42]))
    state = instance.states["stream"]
    assert [finished for _, finished in state.generator.calls] == [False, True]


def test_streaming_talker_rejects_request_when_factory_feature_is_disabled() -> None:
    instance = scheduler(enabled=False)

    with pytest.raises(RuntimeError, match="early speech is disabled"):
        instance.on_streaming_new_request("stream", payload())
