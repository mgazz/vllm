# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 IBM.
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
"""Inference-only IBM/NASA Prithvi Geospatial model."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, Union

import torch
import torch.nn as nn
from terratorch.vllm import (DummyDataGenerator, InferenceRunner,
                             MultiModalDataGenerator)
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    IsAttentionFree, MultiModalEmbeddings, SupportsMultiModalWithRawInput,
    default_pooling_type)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalFieldElem, MultiModalInputs,
                                    MultiModalKwargsItem,
                                    MultiModalKwargsItems,
                                    MultiModalSharedField, PlaceholderRange)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors


class TerratorchProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class TerratorchInputBuilder(BaseDummyInputsBuilder[TerratorchProcessingInfo]):

    def __init__(self, info: TerratorchProcessingInfo):
        super().__init__(info)
        self.dummy_data_generator = DummyDataGenerator(
            self.info.get_hf_config().to_dict()["pretrained_cfg"])

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        # This model input is fixed and is in the form of a torch Tensor.
        # The size of pixel_values might change in the cases where we resize
        # the input but never exceeds the dimensions below.
        return self.dummy_data_generator.get_dummy_mm_data()


class TerratorchMultiModalProcessor(BaseMultiModalProcessor):

    def __init__(
            self,
            info: TerratorchProcessingInfo,
            dummy_inputs: "BaseDummyInputsBuilder[TerratorchProcessingInfo]",
            *,
            cache: Optional[ProcessingCache] = None) -> None:

        super().__init__(info=info, dummy_inputs=dummy_inputs, cache=cache)
        self.mm_data_generator = MultiModalDataGenerator(
            self.info.get_hf_config().to_dict()["pretrained_cfg"])

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = self.mm_data_processor._get_mm_fields_config()
        mm_fields_config = {}
        for field_name, field_modality in fields.item():
            mm_fields_config[field_name] = MultiModalFieldConfig.shared(
                batch_size=1, modality=field_modality)
        return mm_fields_config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        mm_kwargs = {}

        for k, v in mm_data.items():
            if isinstance(v, dict) and k == "image":
                mm_kwargs.update(v)
            else:
                mm_kwargs[k] = v
        mm_placeholders = {"image": [PlaceholderRange(offset=0, length=0)]}

        # This model receives in input a multi-dimensional tensor representing
        # a single image patch and therefore it is not to be split
        # into multiple elements, but rather to be considered a single one.
        # Hence, the decision of using a MultiModalSharedField.
        # The expected shape is (num_channels, width, height).

        # This model however allows the user to also submit multiple image
        # patches as a batch, adding a further dimension to the above shape.
        # At this stage we only support submitting one patch per request and
        # batching is achieved via vLLM batching.
        # TODO (christian-pinto): enable support for multi patch requests
        # in tandem with vLLM batching.
        multimodal_kwargs_items = [
            MultiModalKwargsItem.from_elems([
                MultiModalFieldElem(
                    modality="image",
                    key=key,
                    data=data,
                    field=MultiModalSharedField(1),
                ) for key, data in mm_kwargs.items()
            ])
        ]

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=[1],
            mm_kwargs=MultiModalKwargsItems.from_seq(multimodal_kwargs_items),
            mm_hashes=None,
            mm_placeholders=mm_placeholders,
        )


@default_pooling_type("All")
@MULTIMODAL_REGISTRY.register_processor(
    TerratorchMultiModalProcessor,
    info=TerratorchProcessingInfo,
    dummy_inputs=TerratorchInputBuilder,
)
class Terratorch(nn.Module, IsAttentionFree, SupportsMultiModalWithRawInput):
    """Prithvi Masked Autoencoder"""

    is_pooling_model = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]

        self.inference_runner = InferenceRunner(config)
        self.model = self.inference_runner.model

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {"encode": Pooler.for_encode(pooler_config)}, )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        # We do not really use any input tokens and therefore no embeddings
        # to be calculated. However, due to the mandatory token ids in
        # the input prompt we pass one token and the size of the dummy
        # embedding tensors must reflect that.
        return torch.empty((input_ids.shape[0], 0))

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        model_output = self.inference_runner.forward(**kwargs)

        return model_output.output

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_list = []
        model_buffers = dict(self.named_buffers())
        loaded_buffers = []
        for key, value in weights:
            if type(value) is dict or isinstance(value, OrderedDict):
                if key == "state_dict":
                    weights_to_parse = value
                    for name, weight in weights_to_parse.items():
                        name = f"inference_runner.{name}"

                        if "pos_embed" in name:
                            continue

                        if "_timm_module." in name:
                            name = name.replace("_timm_module.", "")

                        # this model requires a couple of buffers to be loaded
                        # that are not loadable with the AutoWeightsLoader
                        if name in model_buffers:
                            if "_timm_module." in name:
                                name = name.replace("_timm_module.", "")
                            buffer = model_buffers[name]
                            weight_loader = getattr(buffer, "weight_loader",
                                                    default_weight_loader)
                            weight_loader(buffer, weight)
                            loaded_buffers.append(name)
                        else:
                            params_list.append((name, weight))
                    break

            elif isinstance(value,
                            torch.Tensor):  # To handle WxCDownscalingTask
                params_list.append((f"inference_runner.model.{key}", value))

        # Load the remaining model parameters
        loader = AutoWeightsLoader(self, )
        autoloaded_weights = loader.load_weights(params_list)

        return autoloaded_weights.union(set(loaded_buffers))
