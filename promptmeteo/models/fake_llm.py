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
from typing import Any
from typing import List
from typing import Dict
from typing import Mapping
from typing import Optional

from langchain.llms.base import LLM
from langchain.embeddings import FakeEmbeddings

from .base import BaseModel


class FakeStaticLLM(LLM):

    """
    Fake Static LLM wrapper for testing purposes.
    """

    response:str='positive'

    @property
    def _llm_type(
        self
    ) -> str:

        """
        Return type of llm.
        """

        return "fake-static"


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:

        """
        Return static response.
        """

        return self.response


    @property
    def _identifying_params(
        self
    ) -> Mapping[str, Any]:

        return {}


class FakePromptCopyLLM(LLM):

    """
    Fake Prompt Copy LLM wrapper for testing purposes.
    """


    @property
    def _llm_type(self) -> str:

        """
        Return type of llm.
        """

        return "fake-prompt-copy"


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:

        """
        Return prompt.
        """

        return prompt


    @property
    def _identifying_params(
        self
    ) -> Mapping[str, Any]:

        return {}


class FakeListLLM(LLM):

    """
    Fake LLM wrapper for testing purposes.
    """

    responses: List =['uno','dos','tres']
    i: int = 0


    @property
    def _llm_type(self) -> str:

        """
        Return type of llm.
        """

        return "fake-list"


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:

        """
        First try to lookup in queries, else return 'foo' or 'bar'.
        """

        response = self.responses[self.i]
        self.i += 1

        return response


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class ModelNames(Enum):

    MODEL_1 = 'fake_static'
    MODEL_2 = 'fake_prompt_copy'
    MODEL_3 = 'fake_list'


class FakeLLM(BaseModel):

    def __init__(
        self,
        model_name           : Optional[str] = '',
        model_params         : Optional[Dict] = {},
        model_provider_token : Optional[str] = '',
    ) -> None:

        self._embeddings = FakeEmbeddings(size=64)

        if model_name == ModelNames.MODEL_1.value:
            self._llm = FakeStaticLLM()

        elif model_name == ModelNames.MODEL_2.value:
            self._llm = FakePromptCopyLLM()

        elif model_name == ModelNames.MODEL_3.value:
            self._llm = FakeListLLM()

        else:
            raise ValueError(
                "{model_name} is not in the list of supported FakeLLMS: "
                "[i.value for i in ModelNames]"
                )

