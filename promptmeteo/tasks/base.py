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
from typing import Dict
from typing import Optional
from typing_extensions import Self

from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

from promptmeteo.models import ModelFactory, BaseModel
from promptmeteo.prompts import PromptFactory, BasePrompt
from promptmeteo.parsers import ParserFactory, BaseParser
from promptmeteo.selector import SelectorFactory, BaseSelector


class BaseTask():

    """
    Base Task interface.
    """

    def __init__(
        self,
        verbose : bool = False
    ):

        self._model    = None
        self._parser   = None
        self._prompt   = None
        self._selector = None
        self._verbose  = verbose


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

        if self._verbose:
            print('\n\nPROMPT INPUT\n\n', prompt)

        result = self.model.run(prompt)

        if self._verbose:
            print('\n\nMODEL OUTPUT\n\n', result)

        result = self.parser.run(result)

        return result


class BaseTaskBuilder():

    """
    Builder of Tasks.
    """

    BASE_PROMPT = None

    def __init__(
        self,
        verbose     : bool,
    ) -> None:

        self._labels = None
        self._task = BaseTask(verbose=verbose)


    @property
    def task(self) -> BaseTask:
        """Task to built."""
        return self._task


    def build_prompt(
        self,
        prompt_domain : str,
        prompt_labels : List[str],
        prompt_detail = str,
    ) -> Self:

        """
        Builds a the prompt for the task.
        """

        self._task.prompt = PromptFactory.factory_method(
            prompt_type   = self.BASE_PROMPT,
            prompt_domain = prompt_domain,
            prompt_labels = prompt_labels,
            prompt_detail = prompt_detail,
        )

        return self


    def build_selector_by_train(
        self,
        examples        : List[str],
        annotations     : List[str],
        selector_k           : int,
        selector_algorithm   : str,
    ) -> Self:

        """
        Builds a the selector for the task by training a new selector.
        """

        if not self._task.model:
            raise RuntimeError(
                'Selector algorithm is trying yo be built but there is no'
                'LLM model or embeddings loaded. You need to call function'
                '`build_model()` before calling `build_selector()`.'
            )

        if not self._task.model.embeddings:
            raise RuntimeError(
                'Selector algorithm is trying yo be built but there is no'
                'embeddigns for model {self._task._model.__name__}.'
            )

        embeddings = self._task.model.embeddings

        self._task.selector = SelectorFactory.factory_method(
            embeddings           = embeddings,
            selector_k           = selector_k,
            selector_algorithm   = selector_algorithm
        ).train(
            examples             = examples,
            annotations          = annotations,
        )

        return self


    def build_model(
        self,
        model_name           : str = '',
        model_provider_name  : str = '',
        model_provider_token : str = '',
        model_params         : Dict = {}
    ) -> Self:

        """
        Builds a the model for the task.
        """

        self._task.model = ModelFactory.factory_method(
            model_name = model_name,
            model_provider_name = model_provider_name,
            model_provider_token = model_provider_token,
            model_params = model_params
        )

        return self


    def build_parser(
        self,
        parser_type              : str,
        prompt_labels            : List[str],
    ) -> Self:

        """
        Builds a the parser for the task.
        """

        self._task.parser = ParserFactory.factory_method(
            parser_type=parser_type,
            prompt_labels=prompt_labels,
            prompt_labels_separator=','
        )

        return self


    def build_selector_by_load(
        self,
        model_path         : str,
        selector_k         : str,
        selector_algorithm : str
    ) -> Self:

        """
        Builds a the selector for the task by loading a pretrained selector.
        """

        if not self._task.model:
            raise RuntimeError(
                'Selector algorithm is trying to be built but there is no'
                'LLM model or embeddings loaded. You need to call function'
                '`build_model()` before calling `load_selector()`.'
            )

        if not self._task.model.embeddings:
            raise RuntimeError(
                'Selector algorithm is trying to be built but there is no'
                'embeddigns for model {self._task._model.__name__}.'
            )

        embeddings = self._task.model.embeddings

        self._task.selector = SelectorFactory.factory_method(
            embeddings           = embeddings,
            selector_k           = selector_k,
            selector_algorithm   = selector_algorithm
        ).load_example_selector(
            model_path           = model_path
        )

        return self
