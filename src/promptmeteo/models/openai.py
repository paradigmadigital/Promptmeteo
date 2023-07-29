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

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from .base import BaseModel


class ModelNames(str,Enum):

    MODEL_1 = 'text-davinci-003'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ModelParams(Enum):

    class MODEL_1:
        model_task   = "text2text-generation"
        model_kwargs ={"temperature": 0.7, "max_tokens": 256, "max_retries": 3}


class OpenAILLM(BaseModel):


    def __init__(
        self,
        model_name           : Optional[str] = '',
        model_params         : Optional[Dict] = {},
        model_provider_token : Optional[str] = '',
    ) -> None:

        """
        """

        if not ModelNames.has_value(model_name):
            raise ValueError(
                f'`model_name`={model_name} not in supported model names: '
                f'{list(ModelNames.__members__.keys())}')

        if not model_params:
            model_params = ModelParams[ModelNames(model_name).name].value

        self._llm = OpenAI(
            model_name=model_name,
            openai_api_key=model_provider_token,
            #openai_organization="YOUR_ORGANIZATION_ID"
            )

        self._embeddings = OpenAIEmbeddings(
            openai_api_key=model_provider_token
        )

