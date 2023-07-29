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


class ModelProvider(str, Enum):

    """
    LLM providers currently supported by Promptmeteo
    """

    PROVIDER_0 = "fake_llm"
    PROVIDER_1 = "openai"
    PROVIDER_2 = "hf_hub_api"
    PROVIDER_3 = "hf_pipeline"


class ModelFactory():

    """
    The ModelFactory class is used to create a BaseModel object from the given
    AutoLabelConfig configuration.
    """

    @staticmethod
    def factory_method(
        model_name           : str,
        model_provider_name  : str,
        model_provider_token : str,
        model_params         : Dict
    ) -> BaseModel:

        """
        Returns a BaseModel object configured with the settings found in the
        provided AutolabelConfig.

        Args:
            model_provider_name:
            model_name:
        Returns:
            model: a fully configured BaseModel object
        """

        if model_provider_name == ModelProvider.PROVIDER_0.value:
            from .fake_llm import FakeLLM
            model_cls = FakeLLM

        elif model_provider_name == ModelProvider.PROVIDER_1.value:
            from .openai import OpenAILLM
            model_cls = OpenAILLM

        elif model_provider_name == ModelProvider.PROVIDER_2.value:
            from .hf_hub_api import HFHubApiLLM
            model_cls = HFHubApiLLM

        elif model_provider_name == ModelProvider.PROVIDER_3.value:
            from .hf_pipeline import HFPipelineLLM
            model_cls = HFPipelineLLM

        else:
            raise ValueError(
                f'{model_provider_name} is not in the list of supported '
                f'providers: {[i.value for i in ModelProvider]}'
            )

        return model_cls(
            model_name=model_name,
            model_params=model_params,
            model_provider_token=model_provider_token,
        )
