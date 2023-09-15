#!/usr/bin/python3

#  Copyright (c) 2023 Paradigma Digital S.L.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from enum import Enum
from typing import Dict

from .base import BaseModel
from .openai import OpenAILLM
from .fake_llm import FakeLLM
from .hf_hub_api import HFHubApiLLM
from .hf_pipeline import HFPipelineLLM
from .google_vertexai import GoogleVertexAILLM


class ModelProvider(str, Enum):

    """
    LLM providers currently supported by Promptmeteo
    """

    PROVIDER_0: str = "fake-llm"
    PROVIDER_1: str = "openai"
    PROVIDER_2: str = "hf_hub_api"
    PROVIDER_3: str = "hf_pipeline"
    PROVIDER_4: str = "google-vertexai"



class ModelFactory:

    """
    The ModelFactory class is used to create a BaseModel object from the given
    configuration.
    """

    MAPPING = {
        ModelProvider.PROVIDER_0: FakeLLM,
        ModelProvider.PROVIDER_1: OpenAILLM,
        ModelProvider.PROVIDER_2: HFHubApiLLM,
        ModelProvider.PROVIDER_3: HFPipelineLLM,
        ModelProvider.PROVIDER_3: GoogleVertexAILLM,
    }

    @classmethod
    def factory_method(
        cls,
        model_name: str,
        model_provider_name: str,
        model_provider_token: str,
        model_params: Dict,
    ) -> BaseModel:
        """
        Returns a BaseModel object configured with the settings found in the
        provided parameters.
        """
        model_cls = cls.MAPPING.get(model_provider_name)

        if model_provider_name == ModelProvider.PROVIDER_0.value:
            model_cls = FakeLLM

        elif model_provider_name == ModelProvider.PROVIDER_1.value:
            model_cls = OpenAILLM

        elif model_provider_name == ModelProvider.PROVIDER_2.value:
            model_cls = HFHubApiLLM

        elif model_provider_name == ModelProvider.PROVIDER_3.value:
            model_cls = HFPipelineLLM
        
        elif model_provider_name == ModelProvider.PROVIDER_4.value:
            model_cls = GoogleVertexAILLM

        else:
            raise ValueError(
                f"{cls.__name__} error in `factory_method()`. "
                f"{model_provider_name} is not in the list of supported "
                f"providers: {[i.value for i in ModelProvider]}"
            )

        return model_cls(
            model_name=model_name,
            model_params=model_params,
            model_provider_token=model_provider_token,
        )
