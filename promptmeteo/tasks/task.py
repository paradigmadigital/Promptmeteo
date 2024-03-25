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

from ..models import BaseModel
from ..prompts import BasePrompt
from ..parsers import BaseParser
from ..selector import (
    BaseSelector,
    BaseSelectorSupervised,
    BaseSelectorUnsupervised,
)


class Task:
    """
    Base Task interface.
    """

    def __init__(
        self,
        language: str,
        task_type: str,
        verbose: bool = False,
    ):
        self._model = None
        self._parser = None
        self._prompt = None
        self._selector = None
        self._verbose = verbose
        self._language = language
        self._task_type = task_type

    # Getters
    @property
    def language(
        self,
    ) -> str:
        """
        Get Task Language.
        """
        return self._language

    @property
    def task_type(
        self,
    ) -> str:
        """
        Get Task type.
        """
        return self._task_type

    @property
    def prompt(
        self,
    ) -> BasePrompt:
        """
        Get Task Prompt.
        """
        return self._prompt

    @property
    def model(
        self,
    ) -> BaseModel:
        """
        Get Task Model.
        """
        return self._model

    @property
    def selector(
        self,
    ) -> BaseSelector:
        """
        Get Task Selector.
        """
        return self._selector

    @property
    def parser(
        self,
    ) -> BaseParser:
        """
        Get Task Parser.
        """
        return self._parser

    # Setters
    @prompt.setter
    def prompt(
        self,
        prompt: BasePrompt,
    ) -> None:
        """
        Set Task Prompt.
        """
        self._prompt = prompt

    @model.setter
    def model(
        self,
        model: BaseModel,
    ) -> None:
        """
        Set Task Model.
        """
        self._model = model

    @selector.setter
    def selector(
        self,
        selector: BaseSelector,
    ) -> None:
        """
        Set Task Selector.
        """
        self._selector = selector

    @parser.setter
    def parser(
        self,
        parser: BaseParser,
    ) -> None:
        """
        Task Parser
        """
        self._parser = parser

    def _get_prompt(
        self,
        sample: str,
    ) -> str:
        """
        Create a PipelinePromptTemplate by merging the PromptTemplate and the
        FewShotPromptTemplate.
        """

        intro_prompt = self.prompt.run()

        if isinstance(self.selector, BaseSelectorSupervised):
            examples = self.selector.run(sample)
        elif isinstance(self.selector, BaseSelectorUnsupervised):
            examples = self.selector.run()
        else:
            examples = ""

        variables = dict(__SAMPLE__=sample, __EXAMPLES__=examples)

        final_prompt = intro_prompt.format(
            **{
                k: v
                for (k, v) in variables.items()
                if k in intro_prompt.input_variables
            }
        )

        return final_prompt

    def run(
        self,
        example: str,
    ) -> str:
        """
        Given a text sample, return the text predicted by Promptmeteo.
        """

        sample = example.replace("{", "{{").replace("}", "}}")

        prompt = self._get_prompt(sample)
        if self._verbose:
            print("\n\nPROMPT INPUT\n\n", prompt)

        output = self.model.run(prompt)
        if self._verbose:
            print("\n\nMODEL OUTPUT\n\n", output)

        result = self.parser.run(output)
        if self._verbose:
            print("\n\nPARSE RESULT\n\n", result)

        return result
