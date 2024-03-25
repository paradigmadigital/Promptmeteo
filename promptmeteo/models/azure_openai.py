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

import os
from enum import Enum
from typing import Dict
from typing import Optional

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI

from .base import BaseModel


class ModelTypes(str, Enum):
    """
    Enum of available model types.
    """

    GPT35TurboInstruct: str = "gpt-3.5-turbo-instruct"
    GPT35Turbo: str = "gpt-3.5-turbo-16k"

    @classmethod
    def has_value(
        cls,
        value: str,
    ) -> bool:
        """
        Checks if the value is in the enum or not.
        """

        return value in cls._value2member_map_


class ModelEnum(Enum):
    """
    Model types with their parameters.
    """

    class GPT35TurboInstruct:
        """
        Default parameters for TextDavinci003 model.
        """

        client = AzureOpenAI
        embedding = OpenAIEmbeddings

        params = {
            "model_name": "gpt-3.5-turbo-instruct",
            "temperature": 0.0,
            # "max_tokens": 4000,
            "max_retries": 3,
        }

    class GPT35Turbo:
        """
        Default parameters for GPT35Turbo model.
        """

        client = AzureChatOpenAI
        embedding = OpenAIEmbeddings

        params: dict = {
            "deployment_name": "gpt-3.5-turbo-16k",
            "temperature": 0.0,
            # "max_tokens": 16384,
            "max_retries": 3,
        }


class AzureOpenAILLM(BaseModel):
    """
    OpenAI LLM model.
    """

    def __init__(
        self,
        model_name: Optional[str] = "",
        model_params: Optional[Dict] = None,
        model_provider_token: Optional[str] = "",
        **kwargs,
    ) -> None:
        """
        Make predictions using a model from OpenAI.
        It will use the os environment called OPENAI_ORGANIZATION for instance
        the LLM
        """

        if not ModelTypes.has_value(model_name):
            raise ValueError(
                f"`model_name`={model_name} not in supported model names: "
                f"{[i.value for i in ModelTypes]}"
            )

        super(AzureOpenAILLM, self).__init__()

        # Model name
        model = ModelTypes(model_name).name

        # Model Parameters
        if not model_params:
            model_params = (
                ModelEnum[model].value.params
                if not model_params
                else model_params
            )
        self.model_params = model_params

        # Model
        self._llm = ModelEnum[model].value.client(
            openai_api_key=model_provider_token, **self.model_params
        )

        # Embeddings
        self._embeddings = ModelEnum[model].value.embedding(
            deployment="text-embedding-ada-002-v2",
            openai_api_key=os.environ.get("EMBEDDINGS_API_KEY", None)
            or model_provider_token,
            openai_api_base=os.environ.get("EMBEDDINGS_API_BASE", ""),
        )
