# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class RequestLogger:
    def __init__(
        self, *, max_log_len: int | None, log_prompt: bool = False
    ) -> None:
        self.max_log_len = max_log_len
        self.log_prompt = log_prompt

        if not logger.isEnabledFor(logging.INFO):
            logger.warning_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than INFO. "
                "No request information will be logged."
            )
        elif not logger.isEnabledFor(logging.DEBUG) and not self.log_prompt:
            logger.info_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than DEBUG. "
                "Only limited information will be logged to minimize overhead. "
                "To view more details, set `VLLM_LOGGING_LEVEL=DEBUG` "
                "or `--enable-log-prompt`."
            )

    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        prompt_to_log = prompt
        token_ids_to_log = prompt_token_ids
        if self.max_log_len is not None:
            if prompt_to_log is not None:
                prompt_to_log = prompt_to_log[: self.max_log_len]
            if token_ids_to_log is not None:
                token_ids_to_log = token_ids_to_log[: self.max_log_len]

        if self.log_prompt:
            logger.info("Request %s prompt: %r", request_id, prompt_to_log)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Request %s details: prompt: %r, "
                "prompt_token_ids: %s, "
                "prompt_embeds shape: %s.",
                request_id,
                prompt_to_log,
                token_ids_to_log,
                prompt_embeds.shape if prompt_embeds is not None else None,
            )

        logger.info(
            "Received request %s: params: %s, lora_request: %s.",
            request_id,
            params,
            lora_request,
        )

    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Sequence[int] | None,
        finish_reason: str | None = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None and outputs is not None:
            outputs = outputs[:max_log_len]

        stream_info = ""
        if is_streaming:
            stream_info = " (streaming delta)" if delta else " (streaming complete)"

        if logger.isEnabledFor(logging.DEBUG):
            if max_log_len is not None and output_token_ids is not None:
                output_token_ids = list(output_token_ids)[:max_log_len]

            logger.debug(
                "Generated response %s%s details: output_token_ids: %s",
                request_id,
                stream_info,
                output_token_ids,
            )

        logger.info(
            "Generated response %s%s: output: %r, finish_reason: %s",
            request_id,
            stream_info,
            outputs,
            finish_reason,
        )
