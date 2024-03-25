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
from typing import Optional
import os
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import HuggingFaceEmbeddings

from .base import BaseModel


class ModelTypes(str, Enum):
    """
    Enum of available model types.
    """

    AnthropicClaudeV2: str = "anthropic.claude-v2"

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
    Model Parameters.
    """

    class AnthropicClaudeV2:
        """
        Default parameters for Anthropic Claude V2
        """

        client = Bedrock
        embedding = HuggingFaceEmbeddings
        model_task: str = "text2text-generation"
        params: dict = {
            "max_tokens_to_sample": 2048,
            "temperature": 0.3,
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["Human:"],
        }


class BedrockLLM(BaseModel):
    """
    Bedrock LLM model.
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
        It will use the os environment called OPENAI_ORGANIZATION for instance the LLM
        """

        if not ModelTypes.has_value(model_name):
            raise ValueError(
                f"`model_name`={model_name} not in supported model names: "
                f"{[i.name for i in ModelTypes]}"
            )
        self.boto3_bedrock = boto3.client("bedrock-runtime", **kwargs)
        super(BedrockLLM, self).__init__()

        # Model name
        model = ModelTypes(model_name).name

        # Model parameters
        if not model_params:
            model_params = (
                ModelEnum[model].value.params
                if not model_params
                else model_params
            )
        self.model_params = model_params

        # Model
        self._llm = ModelEnum[model].value.client(
            model_id=model_name,
            model_kwargs=self.model_params,
            client=self.boto3_bedrock,
        )

        embedding_name = "sentence-transformers/all-MiniLM-L6-v2"
        if os.path.exists("/home/models/all-MiniLM-L6-v2"):
            embedding_name = "/home/models/all-MiniLM-L6-v2"

        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
