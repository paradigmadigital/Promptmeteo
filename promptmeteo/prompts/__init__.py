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
from typing import List

from .base import BasePrompt


class ClassificationPrompt(BasePrompt):
    pass


class NerPrompt(BasePrompt):
    pass


class PromptTypes(str, Enum):

    """
    Enum of availables prompts.
    """

    PROMPT_1 = "classification"
    PROMPT_2 = "ner"


def read_prompt(prompt_file: str):

    """
    Reads a prompt file (i.e. `.prompt` extension). 
    """

    path = os.path.join(__file__, os.path.pardir, 'sp', prompt_file)
    with open(os.path.abspath(path), encoding='utf-8') as fin:
        prompt_text = fin.read()

    return prompt_text


class PromptFactory():

    """
    Factory of Prompts
    """

    @staticmethod
    def factory_method(
        prompt_type   : str,
        prompt_domain : str,
        prompt_labels : List[str],
        prompt_detail : str,
    ):

        """
        Returns and instance of a BasePrompt object depending on the
        `prompt_type`.
        """

        if prompt_type == PromptTypes.PROMPT_1.value:
            ClassificationPrompt.PROMPT_EXAMPLE = read_prompt(
                'classification.prompt')

            prompt_cls = ClassificationPrompt

        elif prompt_type == PromptTypes.PROMPT_2.value:
            NerPrompt.PROMPT_EXAMPLE = read_prompt('ner.prompt')
            prompt_cls = NerPrompt

        else:
            raise ValueError(
                f"{prompt_type} is not in the list of supported providers: "
                f"{[i.value for i in PromptTypes]}"
                )

        prompt_cls.read_prompt(prompt_cls.PROMPT_EXAMPLE)

        return prompt_cls(
            prompt_domain=prompt_domain,
            prompt_labels=prompt_labels,
            prompt_detail=prompt_detail,
        )
