# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.profiler.event_recorder import emit as emit_profile_event
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:
    import torch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass


logger = logging.getLogger(__name__)


class MiniCPMOThinkerModelRunner(ThinkerModelRunner):
    """Run the thinker and accumulate hidden states for speech conditioning."""

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        enable_talker_cuda_ipc: bool = False,
        enable_talker_start_measurement: bool = False,
        enable_talker_partial_start: bool = False,
        tts_bos_token_id: int | None = None,
        tts_eos_token_id: int | None = None,
    ) -> None:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            get_server_return_hidden_states_mode,
        )

        # note (MayDomine): the thinker initializer requires a nested Qwen config.
        ModelRunner.__init__(self, tp_worker, output_processor)

        # note (MayDomine): parent embedding injection reads these names.
        model = self.model
        self.outer_model = model.thinker
        self.text_model = self.outer_model.model
        self.embed_tokens = self.text_model.embed_tokens
        self.th_host_bufs = None
        self.th_slot = 0
        # note (MayDomine): bound-based injection needs no modality token ids.
        self.image_token_id = -1
        self.video_token_id = -1
        self.audio_token_id = -1

        self.capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if output_processor.capture_hidden
            else get_server_return_hidden_states_mode()
        )
        self.pending_hidden: dict[str, list[torch.Tensor]] = {}
        self.enable_talker_cuda_ipc = bool(enable_talker_cuda_ipc)
        self.enable_talker_start_measurement = bool(enable_talker_start_measurement)
        self.enable_talker_partial_start = bool(enable_talker_partial_start)
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.speech_measurements: dict[str, dict[str, int]] = {}
        self.speech_prompt_has_bos: dict[str, bool] = {}
        if self.enable_talker_start_measurement:
            logger.info(
                "minicpm_talker_start_measurement enabled tts_bos_token_id=%s "
                "tts_eos_token_id=%s",
                self.tts_bos_token_id,
                self.tts_eos_token_id,
            )
        else:
            pass

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Use deployment-wide capture; batch arguments follow the runner interface."""
        return self.capture_hidden_mode

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Use deployment-wide capture; batch arguments follow the runner interface."""
        return self.capture_hidden_mode

    def _prompt_has_tts_bos(self, request_id: str, prompt_ids: object) -> bool:
        cached = self.speech_prompt_has_bos.get(request_id)
        if cached is not None:
            return cached
        else:
            pass
        if hasattr(prompt_ids, "reshape"):
            token_ids = prompt_ids.reshape(-1).tolist()
        else:
            token_ids = list(prompt_ids or [])
        has_tts_bos = (
            self.tts_bos_token_id is not None
            and int(self.tts_bos_token_id) in token_ids
        )
        self.speech_prompt_has_bos[request_id] = has_tts_bos
        return has_tts_bos

    def _record_speech_token(
        self, request_id: str, token_id: int, *, prompt_has_tts_bos: bool
    ) -> None:
        """仅在测量开关开启时记录语音内容出现的关键时刻。"""
        if not self.enable_talker_start_measurement:
            return
        else:
            pass
        if self.tts_bos_token_id is None or self.tts_eos_token_id is None:
            return
        else:
            pass
        if token_id == self.tts_bos_token_id:
            self.speech_measurements[request_id] = {"speech_token_count": 0}
            emit_profile_event(
                request_id=request_id,
                stage="thinker",
                event_name="thinker_speech_content_started",
                metadata={"token_id": token_id, "start_source": "generated_token"},
            )
            return
        else:
            pass
        measurement = self.speech_measurements.get(request_id)
        if measurement is None and prompt_has_tts_bos:
            measurement = {"speech_token_count": 0}
            self.speech_measurements[request_id] = measurement
            emit_profile_event(
                request_id=request_id,
                stage="thinker",
                event_name="thinker_speech_content_started",
                metadata={"token_id": token_id, "start_source": "prompt"},
            )
        else:
            pass
        if measurement is None or measurement.get("speech_ended"):
            return
        else:
            pass
        if token_id == self.tts_eos_token_id:
            measurement["speech_ended"] = 1
            emit_profile_event(
                request_id=request_id,
                stage="thinker",
                event_name="thinker_speech_content_ended",
                metadata={
                    "token_id": token_id,
                    "speech_token_count": measurement["speech_token_count"],
                },
            )
            return
        else:
            pass
        measurement["speech_token_count"] += 1
        count = measurement["speech_token_count"]
        if count in (8, 16, 32):
            emit_profile_event(
                request_id=request_id,
                stage="thinker",
                event_name="thinker_speech_content_ready",
                metadata={"speech_token_count": count},
            )
        else:
            pass

    def _record_speech_finished(self, request_id: str) -> None:
        measurement = self.speech_measurements.pop(request_id, None)
        self.speech_prompt_has_bos.pop(request_id, None)
        if measurement is None:
            return
        else:
            pass
        emit_profile_event(
            request_id=request_id,
            stage="thinker",
            event_name="thinker_speech_generation_finished",
            metadata={
                "speech_token_count": measurement["speech_token_count"],
                "speech_ended": bool(measurement.get("speech_ended")),
            },
        )

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Collect request outputs; the raw result is part of the runner interface."""
        for sched_req in scheduler_output.requests:
            req_output = outputs.get(sched_req.request_id)
            if req_output is None:
                continue
            else:
                pass
            if sched_req.data.req.inflight_middle_chunks > 0:
                continue
            else:
                pass
            if req_output.data is not None:
                self._record_speech_token(
                    sched_req.request_id,
                    int(req_output.data),
                    prompt_has_tts_bos=self._prompt_has_tts_bos(
                        sched_req.request_id, sched_req.data.input_ids
                    ),
                )
            else:
                pass
            if req_output.extra is None:
                continue
            else:
                pass
            hidden = req_output.extra.pop("hidden_states", None)
            if hidden is None:
                continue
            else:
                pass
            hidden = hidden.reshape(-1, hidden.shape[-1])[-1]
            captured_hidden = hidden.detach().clone()
            seq = self.pending_hidden.setdefault(sched_req.request_id, [])
            # note (MayDomine): CUDA graph replay overwrites the original hidden buffer.
            seq.append(captured_hidden)
            if self.enable_talker_partial_start:
                req_output.extra["talker_stream_hidden"] = captured_hidden
            else:
                pass

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        """Do not advance generation state for non-final prefill chunks."""
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if sched_req.data.req.inflight_middle_chunks > 0
        }

    def on_request_finished(
        self, request_id: str, req_data: SGLangARRequestData
    ) -> None:
        """Flush the request's hidden accumulator with a single D2H copy."""
        import torch

        self._record_speech_finished(request_id)
        seq = self.pending_hidden.pop(request_id, None)
        if not seq:
            return
        else:
            pass
        stacked = torch.stack(seq)
        if self.enable_talker_cuda_ipc:
            # 同卡 IPC 路径传递单个连续 GPU tensor，避免逐行导出 IPC handle。
            req_data.extra_model_outputs["hidden_states_seq"] = stacked
        else:
            req_data.extra_model_outputs["hidden_states_seq"] = list(
                stacked.to("cpu").unbind(0)
            )

    def reset_request(self, request_id: str) -> None:
        """Drop accumulated hidden states on abort (no terminal flush runs)."""
        self.pending_hidden.pop(request_id, None)
        self.speech_measurements.pop(request_id, None)
        self.speech_prompt_has_bos.pop(request_id, None)
