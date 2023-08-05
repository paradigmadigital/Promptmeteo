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

from typing import List

import yaml
from langchain import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate


class BasePrompt():

    """
    Prompt class interface.
    """

    def __init__(
        self,
        prompt_domain : str = '',
        prompt_labels : str = '',
        prompt_detail : str = '',
    ) -> None:

        """
        Build a Prompt object string given a concrete especification
        """

        self._domain = prompt_domain
        self._labels = prompt_labels

        # Labels
        prompt_labels = ', '.join(prompt_labels)
        prompt_labels = self.PROMPT_LABELS.format(
            __LABELS__=prompt_labels) if prompt_labels else ''

        # Domain
        prompt_domain = self.PROMPT_DOMAIN.format(
            __DOMAIN__=prompt_domain) if prompt_domain else ''

        # Answer format
        answer_format = self.ANSWER_FORMAT

        # Chain of thoughts
        chain_thought = self.CHAIN_THOUGHT

        # Build prompt
        self._prompt = PipelinePromptTemplate(
            final_prompt=PromptTemplate.from_template(
                self.TEMPLATE),
            pipeline_prompts=[
                ('__PROMPT_DOMAIN__',
                    PromptTemplate.from_template(prompt_domain)),
                ('__PROMPT_LABELS__',
                    PromptTemplate.from_template(prompt_labels)),
                ('__ANSWER_FORMAT__',
                    PromptTemplate.from_template(answer_format)),
                ('__CHAIN_THOUGHT__',
                    PromptTemplate.from_template(chain_thought))])


    @property
    def domain(self) -> str:
        """Prompt Domain."""
        return self._domain


    @property
    def labels(self) -> List[str]:
        """Prompt Labels."""
        return self._labels


    @property
    def template(self) -> str:
        """Prompt Template."""
        return self._prompt.format()


    @classmethod
    def read_prompt(cls, prompt_text : str) -> None:

        """
        Reads a Promptmeteo prompt string to build the Task Prompt. Promptmeteo
        prompts are expected to follow the following template:


        Parameters
        ----------

        prompt_text : str


        Returns
        -------

        Self

        """

        try:

            prompt = yaml.load(
                prompt_text,
                Loader=yaml.FullLoader)

            cls.TEMPLATE      = prompt['TEMPLATE']
            cls.PROMPT_DOMAIN = prompt['PROMPT_DOMAIN']
            cls.PROMPT_LABELS = prompt['PROMPT_LABELS']
            cls.ANSWER_FORMAT = prompt['ANSWER_FORMAT']
            cls.CHAIN_THOUGHT = prompt['CHAIN_THOUGHT']

            cls.PROMPT_EXAMPLE = prompt_text

        except Exception as error:
            raise ValueError(
                f'`cls.__name__ class error. `read_prompt` is trying to read'
                f'prompt with a wrong prompt template format. The expected '
                f'string input should be like:\n\n{cls.PROMPT_EXAMPLE}'
                ) from error


    def run(
        self
    ) -> PromptTemplate:

        """
        Returns the prompt template for the current task.
        """

        return self._prompt
