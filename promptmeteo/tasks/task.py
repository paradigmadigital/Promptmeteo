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

from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

from ..models import BaseModel
from ..prompts import BasePrompt
from ..parsers import BaseParser
from ..selector import BaseSelector


class Task():

    """
    Base Task interface.
    """

    def __init__(
        self,
        task_type : str,
        verbose : bool = False
    ):

        self._model    = None
        self._parser   = None
        self._prompt   = None
        self._selector = None
        self._verbose  = verbose
        self._task_type = task_type

    @property
    def prompt(self) -> BasePrompt:
        """Get Task Prompt."""
        return self._prompt

    # Getters
    @property
    def model(self) -> BaseModel:
        """Get Task Model."""
        return self._model

    @property
    def selector(self) -> BaseSelector:
        """Get Task Selector."""
        return self._selector

    @property
    def parser(self) -> BaseParser:
        """Task Parser"""
        return self._parser

    # Setters
    @prompt.setter
    def prompt(self, prompt: BasePrompt) -> None:
        """Set Task Prompt."""
        self._prompt = prompt

    @model.setter
    def model(self, model: BaseModel) -> None:
        """Set Task Model."""
        self._model = model

    @selector.setter
    def selector(self, selector: BaseSelector) -> None:
        """Set Task Selector."""
        self._selector = selector

    @parser.setter
    def parser(self, parser: BaseParser) -> None:
        """Task Parser"""
        self._parser = parser


    def _get_prompt(
        self,
        example: str
    ) -> PipelinePromptTemplate:

        """
        Create a PipelinePromptTemplate by merging the PromptTemplate and the
        FewShotPromptTemplate.
        """

        intro_prompt  = self.prompt.run()

        examples_prompt = self.selector.run() if self.selector else \
            PromptTemplate.from_template('La frase de entrada es: {__INPUT__}')

        return PipelinePromptTemplate(
            final_prompt = PromptTemplate.from_template(
                '''
                {__INSTRUCTION__}

                {__EXAMPLES__}
                '''.replace(' '*4,''
                  ).replace('\n\n','|'
                  ).replace('\n',' '
                  ).replace('|','\n\n')),
            pipeline_prompts=[
                ('__INSTRUCTION__', intro_prompt),
                ('__EXAMPLES__', examples_prompt)]
            ).format(__INPUT__=example)


    def run(
        self,
        example : str
    ) -> str:

        """
        Given a text sample, return the text predicted by Promptmeteo.
        """

        prompt = self._get_prompt(example)
        result = self.model.run(prompt)

        if self._verbose:
            print('\n\nPROMPT INPUT\n\n', prompt)
            print('\n\nMODEL OUTPUT\n\n', result)

        result = self.parser.run(result)

        return result
