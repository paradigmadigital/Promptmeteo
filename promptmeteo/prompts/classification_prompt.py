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
import yaml
from typing import List

from .base import BasePrompt


class ClassificationPrompt(BasePrompt):

    with open(os.path.abspath(
        os.path.join(
            __file__,
            os.path.pardir,
            'templates',
            'sp',
            'classification_prompt.yml')
    )) as fin:

        PROMPT_EXAMPLE = fin.read()


    def __init__(
        self,
        prompt_labels            : str = '',
        prompt_task_info         : str = '',
        prompt_answer_format     : str = '',
        prompt_chain_of_thoughts : str = '',
    ) -> None:

        super().__init__(
            prompt_labels,
            prompt_task_info,
            prompt_answer_format,
            prompt_chain_of_thoughts
        )
