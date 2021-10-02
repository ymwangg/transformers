# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Benchmarking the library on inference and training in PyTorch.
"""


import timeit
from typing import Callable, Optional
from torch_xla.amp import syncfree
from torch import optim

from ..configuration_utils import PretrainedConfig
from ..file_utils import is_py3nvml_available, is_torch_available, is_torch_tpu_available
from ..models.auto.modeling_auto import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from ..utils import logging
from .benchmark_utils import (
    Benchmark,
    Memory,
    MemorySummary,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)


if is_torch_available():
    import torch

    from .benchmark_args import PyTorchBenchmarkArguments

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

if is_py3nvml_available():
    import py3nvml.py3nvml as nvml


logger = logging.get_logger(__name__)


class PyTorchBenchmark(Benchmark):

    args: PyTorchBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "PyTorch"

    @property
    def framework_version(self):
        return torch.__version__

    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_inference)

    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_inference)

    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_train)

    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_train)

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]

        if self.args.torchscript:
            config.torchscript = True

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = config.architectures[0]
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = MODEL_MAPPING[config.__class__](config)

        model.eval()
        model.to(self.args.device)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            #assert self.args.is_gpu, "Mixed precision is possible only for GPU."
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        if self.args.torchscript:
            with torch.no_grad():
                inference_model = torch.jit.trace(model, input_ids)
        else:
            inference_model = model

        def encoder_decoder_forward():
            with torch.no_grad():
                outputs = inference_model(input_ids, decoder_input_ids=input_ids)
            return outputs

        def encoder_forward():
            with torch.no_grad():
                outputs = inference_model(input_ids)
            return outputs

        _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        return _forward

    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        torch.manual_seed(0)
        config = self.config_dict[model_name]

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = config.architectures[0]
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        if self.args.torchscript:
            raise NotImplementedError("Training for torchscript is currently not implemented")
        else:
            if self.args.dump_loss:
                from .model_wrapper import ModelSerializer
                train_model = ModelSerializer(
                    model,
                    f"{model_name}_{'xla' if self.args.is_tpu else 'native'}.pickle"
                )
            else:
                train_model = model

        model.train()
        model.to(self.args.device)

        use_syncfree = self.args.is_tpu and self.args.syncfree
        if self.args.optim == "SGD":
            optim_cls = syncfree.SGD if use_syncfree else optim.SGD
            optimizer = optim_cls(model.parameters(), lr=5e-5, momentum=0.9)
        elif self.args.optim == "Adam":
            optim_cls = syncfree.Adam if use_syncfree else optim.Adam
            optimizer = optim_cls(model.parameters(), lr=5e-5)
        else:
            prefix = "torch_xla.amp.syncfree" if use_syncfree else "torch.optim"
            optim_cls = "{}.{}".format(prefix, self.args.optim)
            raise NotImplementedError(
                "{} has not been implemented".format(optim_cls))

        if self.args.fp16:
            if self.args.is_tpu:
                from torch_xla.amp import autocast, GradScaler
                scaler = syncfree.GradScaler(
                ) if self.args.syncfree else GradScaler()
            else:
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        def compute_loss_and_backprob_encoder():
            optimizer.zero_grad()
            if self.args.fp16:
                with autocast():
                    loss = train_model(input_ids, labels=input_ids)[0]
                scaler.scale(loss).backward()
                if self.args.is_tpu:
                    gradients = xm._fetch_gradients(optimizer)
                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                scaler.step(optimizer)
                scaler.update()
                if self.args.is_tpu:
                    xm.mark_step()
                return loss
            else:
                loss = train_model(input_ids, labels=input_ids)[0]
                loss.backward()
                if self.args.is_tpu:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()
                return loss

        def compute_loss_and_backprob_encoder_decoder():
            optimizer.zero_grad()
            if self.args.fp16:
                with autocast():
                    loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
                scaler.scale(loss).backward()
                if self.args.is_tpu:
                    gradients = xm._fetch_gradients(optimizer)
                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                scaler.step(optimizer)
                scaler.update()
                if self.args.is_tpu:
                    xm.mark_step()
                return loss
            else:
                loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
                loss.backward()
                if self.args.is_tpu:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()
                return loss

        _train = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        return _train

    def _measure_speed(self, func) -> float:
        try:
            if self.args.is_tpu or self.args.torchscript:
                # run additional 10 times to stabilize compilation for tpu and torchscript
                logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                timeit.repeat(
                    func,
                    repeat=1,
                    number=5,
                )

            # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
            runtimes = timeit.repeat(
                func,
                repeat=self.args.repeat,
                number=10,
            )

            if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
                import torch_xla.debug.metrics as met

                self.print_fn(met.metrics_report())

            return min(runtimes) / 10.0
        except RuntimeError as e:
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A"

    def _measure_memory(self, func: Callable[[], None]) -> [Memory, MemorySummary]:
        try:
            if self.args.trace_memory_line_by_line:
                trace = start_memory_tracing("transformers")

            if self.args.is_tpu:
                # tpu
                raise NotImplementedError(
                    "Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `--no-memory` or `args.memory=False`"
                )
            elif self.args.is_gpu:
                if not is_py3nvml_available():
                    logger.warning(
                        "py3nvml not installed, we won't log GPU memory usage. "
                        "Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                    memory = "N/A"
                else:
                    logger.info(
                        "Measuring total GPU usage on GPU device. Make sure to not have additional processes running on the same GPU."
                    )
                    # init nvml
                    nvml.nvmlInit()
                    func()
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                    meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                    max_bytes_in_use = meminfo.used
                    memory = Memory(max_bytes_in_use)
                    # shutdown nvml
                    nvml.nvmlShutdown()
            else:
                # cpu
                memory_bytes = measure_peak_memory_cpu(func)
                memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes

            if self.args.trace_memory_line_by_line:
                summary = stop_memory_tracing(trace)
            else:
                summary = None

            return memory, summary
        except RuntimeError as e:
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A", None
