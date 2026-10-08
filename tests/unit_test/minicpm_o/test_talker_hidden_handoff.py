# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker-to-talker hidden-state handoff contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.comm import stage_io
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import project_thinker_to_talker
from sglang_omni.models.minicpm_o.talker_request import build_talker_request
from sglang_omni.models.minicpm_o import thinker_model_runner
from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner,
)
from sglang_omni.proto.request import StagePayload

TTS_BOS = 100
TTS_EOS = 101


def test_speech_config_keeps_cuda_ipc_opt_in() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    thinker = next(stage for stage in config.stages if stage.name == "thinker")

    assert thinker.factory.enable_talker_cuda_ipc is False
    assert thinker.factory.enable_talker_start_measurement is False


def make_state(hidden_states_seq: object) -> MiniCPMOPipelineState:
    return MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([1, 2])},
        thinker_out={
            "output_ids": [TTS_BOS, 3, 4, TTS_EOS, 9],
            "extra_model_outputs": {"hidden_states_seq": hidden_states_seq},
        },
    )


def test_talker_request_dense_hidden_matches_legacy_list() -> None:
    dense = torch.arange(30, dtype=torch.float32).reshape(6, 5)
    legacy = [row.clone() for row in dense]

    from_legacy = build_talker_request(
        make_state(legacy),
        tts_bos_token_id=TTS_BOS,
        tts_eos_token_id=TTS_EOS,
    )
    from_dense = build_talker_request(
        make_state(dense),
        tts_bos_token_id=TTS_BOS,
        tts_eos_token_id=TTS_EOS,
    )

    assert torch.equal(from_dense["tts_token_ids"], torch.tensor([3, 4]))
    assert torch.equal(from_dense["tts_token_ids"], from_legacy["tts_token_ids"])
    assert torch.equal(from_dense["tts_hidden"], dense[2:4])
    assert torch.equal(from_dense["tts_hidden"], from_legacy["tts_hidden"])


def test_talker_request_dense_hidden_keeps_current_turn_after_history_eos() -> None:
    dense = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([11, TTS_EOS, 12])},
        thinker_out={
            "output_ids": [TTS_BOS, 3, TTS_EOS, TTS_BOS, 7, 8],
            "extra_model_outputs": {"hidden_states_seq": dense},
        },
    )

    result = build_talker_request(
        state,
        tts_bos_token_id=TTS_BOS,
        tts_eos_token_id=TTS_EOS,
    )

    assert torch.equal(result["tts_token_ids"], torch.tensor([7, 8]))
    assert torch.equal(result["tts_hidden"], dense[5:7])


def test_talker_request_dense_hidden_without_tts_span_is_empty() -> None:
    state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([1, 2])},
        thinker_out={
            "output_ids": [3, 4],
            "extra_model_outputs": {"hidden_states_seq": torch.ones(3, 5)},
        },
    )

    result = build_talker_request(
        state,
        tts_bos_token_id=TTS_BOS,
        tts_eos_token_id=TTS_EOS,
    )

    assert result["tts_token_ids"].numel() == 0
    assert result["tts_hidden"].numel() == 0


def test_routing_preserves_dense_hidden_without_tensor_truthiness() -> None:
    dense = torch.ones(4, 5)
    payload = StagePayload(
        request_id="dense-hidden",
        request=None,
        data=make_state(dense).to_dict(),
    )

    projected = project_thinker_to_talker(payload)
    state = MiniCPMOPipelineState.from_dict(projected.data)
    hidden = state.thinker_out["extra_model_outputs"]["hidden_states_seq"]

    assert hidden is dense


def make_runner(*, enable_talker_cuda_ipc: bool) -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.enable_talker_cuda_ipc = enable_talker_cuda_ipc
    runner.enable_talker_start_measurement = False
    runner.tts_bos_token_id = None
    runner.tts_eos_token_id = None
    runner.speech_measurements = {}
    runner.speech_prompt_has_bos = {}
    runner.pending_hidden = {
        "request": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    }
    return runner


def test_speech_measurement_records_start_thresholds_end_and_finish(monkeypatch) -> None:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.enable_talker_start_measurement = True
    runner.tts_bos_token_id = TTS_BOS
    runner.tts_eos_token_id = TTS_EOS
    runner.speech_measurements = {}
    runner.speech_prompt_has_bos = {}
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        thinker_model_runner,
        "emit_profile_event",
        lambda **kwargs: records.append(kwargs),
    )

    runner._record_speech_token(
        "request", TTS_BOS, prompt_has_tts_bos=False
    )
    for token_id in range(32):
        runner._record_speech_token(
            "request", token_id, prompt_has_tts_bos=False
        )
    runner._record_speech_token(
        "request", TTS_EOS, prompt_has_tts_bos=False
    )
    runner._record_speech_finished("request")

    marker_records = records
    assert [record["event_name"] for record in marker_records] == [
        "thinker_speech_content_started",
        "thinker_speech_content_ready",
        "thinker_speech_content_ready",
        "thinker_speech_content_ready",
        "thinker_speech_content_ended",
        "thinker_speech_generation_finished",
    ]
    assert [record["metadata"] for record in marker_records[1:4]] == [
        {"speech_token_count": 8},
        {"speech_token_count": 16},
        {"speech_token_count": 32},
    ]
    assert marker_records[-1]["metadata"] == {
        "speech_token_count": 32,
        "speech_ended": True,
    }


def test_speech_measurement_starts_at_first_token_after_prompt_marker(monkeypatch) -> None:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.enable_talker_start_measurement = True
    runner.tts_bos_token_id = TTS_BOS
    runner.tts_eos_token_id = TTS_EOS
    runner.speech_measurements = {}
    runner.speech_prompt_has_bos = {}
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        thinker_model_runner,
        "emit_profile_event",
        lambda **kwargs: records.append(kwargs),
    )

    runner._record_speech_token("request", 7, prompt_has_tts_bos=True)

    assert runner.speech_measurements["request"] == {"speech_token_count": 1}
    assert [record["event_name"] for record in records] == [
        "thinker_speech_content_started",
    ]
    assert records[-1]["metadata"] == {"token_id": 7, "start_source": "prompt"}


@pytest.mark.parametrize("enable_talker_cuda_ipc", [False, True])
def test_runner_final_hidden_contract(enable_talker_cuda_ipc: bool) -> None:
    runner = make_runner(enable_talker_cuda_ipc=enable_talker_cuda_ipc)
    req_data = SimpleNamespace(extra_model_outputs={})

    runner.on_request_finished("request", req_data)

    hidden = req_data.extra_model_outputs["hidden_states_seq"]
    assert runner.pending_hidden == {}
    if enable_talker_cuda_ipc:
        assert isinstance(hidden, torch.Tensor)
        assert hidden.is_contiguous()
        assert torch.equal(hidden, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    else:
        assert isinstance(hidden, list)
        assert len(hidden) == 2
        assert all(tensor.device.type == "cpu" for tensor in hidden)
        assert torch.equal(torch.stack(hidden), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shm_fallback_materializes_payload_data_on_cpu() -> None:
    payload = StagePayload(
        request_id="shm-fallback",
        request=None,
        data={
            "dense": torch.arange(
                6, dtype=torch.float32, device="cuda"
            ).reshape(2, 3),
            "metadata": {"cpu": torch.tensor([7])},
        },
    )

    materialized = stage_io.materialize_payload_data_on_cpu(payload)

    assert not materialized.data["dense"].is_cuda
    assert torch.equal(
        materialized.data["dense"],
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )
    assert torch.equal(materialized.data["metadata"]["cpu"], torch.tensor([7]))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runner_ipc_contract_keeps_dense_hidden_on_cuda() -> None:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.enable_talker_cuda_ipc = True
    runner.enable_talker_start_measurement = False
    runner.tts_bos_token_id = None
    runner.tts_eos_token_id = None
    runner.speech_measurements = {}
    runner.speech_prompt_has_bos = {}
    runner.pending_hidden = {
        "request": [
            torch.tensor([1.0, 2.0], device="cuda"),
            torch.tensor([3.0, 4.0], device="cuda"),
        ]
    }
    req_data = SimpleNamespace(extra_model_outputs={})

    runner.on_request_finished("request", req_data)

    hidden = req_data.extra_model_outputs["hidden_states_seq"]
    assert isinstance(hidden, torch.Tensor)
    assert hidden.is_cuda
    assert hidden.is_contiguous()
    assert torch.equal(hidden.cpu(), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
