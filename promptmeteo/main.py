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
import tarfile
import tempfile
from typing import List
from typing import Dict
from typing import Optional
from typing_extensions import Self

from promptmeteo.tasks import BaseTask
from promptmeteo.tasks import BaseTaskBuilder
from promptmeteo.tasks import TaskBuilderFactory


class Promptmeteo():


    """
    'Sun is setting on the New Republic. It's time for the ResistencIA to rise'

                                         - Padme Amidala, mother of Leia -
    """


    def __init__(
        self,
        task_type            : str,
        model_name           : str,
        model_provider_name  : str,
        model_params         : Optional[Dict] = {},
        model_provider_token : Optional[str] = None,
        prompt_domain        : Optional[str] = '',
        prompt_labels        : List[str] = [],
        prompt_detail        : Optional[str] = None,
        selector_k           : int = 10,
        selector_algorithm   : str = 'mmr',
        verbose              : bool = False
    ) -> None:

        """
        Prompmeteo is tool, powered by LLMs, which isable to solve NLP models
        such as text classification and Named Entity Recognition. Its interface
        is similar to a conventional ML model, which allows Prometeo to be used
        in a MLOps pipeline easily.

        Parameters
        ----------

        task_type : str

        model_name : str

        model_provider_name : str

        model_provider_token : Optional[str]

        prompt_domain : str

        prompt_labels : List[str]

        prompt_task_info : Optional[str]

        selector_k : int

        selector_algorithm : str

        verbose : bool


        Returns
        -------

        None

        Example
        -------

        >>> from promptmeteo import Promptmeteo

        >>> classifier = Promptmeteo(
        >>>     task_type='classification',
        >>>     model_provider_name='hf_pipeline',
        >>>     model_name='google/flan-t5-small',
        >>>     prompt_labels=['positivo','negativo','neutral'],
        >>> )

        >>> classifier.train(
        >>>     examples = ['estoy feliz', 'me da igual', 'no me gusta'],
        >>>     annotations = ['positivo', 'neutral', 'negativo'],
        >>>     model_name = 'google/flan-t5-small'
        >>>     model_provider_name='hf_pipeline',
        >>>     selector_algorithm='mmr',
        >>>     selector_k=3)

        >>> classifier.predict(['que guay!!'])

        [['positive']]
        """

        self._builder = TaskBuilderFactory.factory_method(
            task_type                = task_type,
            task_labels              = prompt_labels,
            verbose                  = verbose
        )

        # Build model
        self.builder.build_model(
            model_name               = model_name,
            model_provider_name      = model_provider_name,
            model_provider_token     = model_provider_token,
            model_params             = model_params
        )

        # Build prompt
        self.builder.build_prompt(
            prompt_domain = prompt_domain,
            prompt_labels = prompt_labels,
            prompt_detail = prompt_detail,
        )

        # Build parser
        self.builder.build_parser(
            parser_type              = task_type,
            prompt_labels            = prompt_labels,
        )

        # Selector config
        self._selector_k             = selector_k
        self._selector_algorithm     = selector_algorithm


    @property
    def builder(self) -> BaseTaskBuilder:
        """Task Builder."""
        return self._builder


    @property
    def task(self) -> BaseTask:
        """Task."""
        return self._builder.task


    @property
    def is_trained(self) -> bool:
        """Check if Promptmeteo intance is trained."""
        return self._builder.task.selector is not None


    def read_prompt(
        self,
        prompt_text : str
    ) -> Self:
        
        """
        Reads a Promptmeteo prompt string to build the Task Prompt. Promptmeteo
        prompts are expected to follow the following template

        ```
        TEMPLATE:
            "
            Your task is to classify a text in categories:
            {__LABELS__}
            {__TASK_INFO__}
            {__ANSWER_FORMAT__}
            {__CHAIN_OF_THOUGHTS__}
            "

        LABELS:
            ["positive", "negative", "neutral"]

        TASK_INFO:
            "The text is a sentence written by a human and you have to classify
            it in according to its sentiment."

        ANSWER_FORMAT:
            "Your answer must include the name of the category in a unique word
            in lower case and without puntuation."

        CHAIN_OF_THOUGHTS:
            "Please explain your answer step by step before saying the name of
            the category"
        ```

        Parameters
        ----------

        prompt_text : str

        Returns
        -------

        Self

        """

        self.task.prompt.read_prompt(prompt_text)
        self.task.prompt.__init__()

        return self


    def predict(
        self,
        examples : List[str],
    ) -> List[str]:

        """
        Predicts over new text samples.

        Parameters
        ----------

        examples : List[str]

        Returns
        -------

        List[str]

        """


        if not isinstance(examples,list):
            raise ValueError(
               f'Arguments `examples` and `annotations` are expected to be of '
               f'type `List[str]`. Instead they got: examples {type(examples)}'
            )

        if not all([isinstance(val,str) for val in examples]):

            raise ValueError(
               'Arguments `examples` are expected to be of type `List[str]`. '
               'Some values seem no to be of type `str`.'
            )

        results = []
        for example in examples:
            results.append(self.task.run(example))

        return results


    def train(
        self,
        examples    : List[str],
        annotations : List[str]
    ) -> Self:

        """
        Trains the model given examples and its annotations. The training
        process create a vector store with all the training texts in memory

        Parameters
        ----------

        examples : List[str]

        annotations : List[str]


        Returns
        -------

        self

        """


        if (not isinstance(examples,list) or
            not isinstance(annotations,list)):

            raise ValueError(
               f'Arguments `examples` and `annotations` are expected to be of '
               f'type `List[str]`. Instead they got: examples {type(examples)}'
               f' annotations {type(annotations)}.'
            )

        if (not all([isinstance(val,str) for val in examples]) or
            not all([isinstance(val,str) for val in annotations])):

            raise ValueError(
               f'Arguments `examples` and `annotations` are expected to be of '
               f'type `List[str]`. Some values seem no to be of type `str`.'
            )

        if len(examples) != len(annotations):

            raise ValueError(
               f'Arguments `examples` and `annotations` are expected to have '
               f'the same length. examples=({len(examples)},) annotations= '
               f'({len(annotations)},)'
            )

        self.builder.build_selector_by_train(
            examples                 = examples,
            annotations              = annotations,
            selector_k               = self._selector_k,
            selector_algorithm       = self._selector_algorithm
        )

        return self


    def save_model(
        self,
        model_path : str,
    ) -> Self:

        """
        Saves the training result of the model in disk.

        Parameters
        ----------

        model_path : str


        Returns
        -------

        self : Promptmeteo

        """

        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)

        if not self.is_trained:
            raise RuntimeError(
                f'{self.__class__.__name__} error in `save_model()`. '
                f'You are trying to save a model that has non been trained. '
                f'Please, call `train()` function before'
            )

        if not model_name.endswith('.meteo'):
            raise ValueError(
                f'{self.__class__.__name__} error in `save_model()`. '
                f'model_path="{model_path}" has a bad model name extension. '
                f'Model name must end with `.meteo` (i.e. `./model.meteo`)'
            )

        if model_dir!='' and not os.path.exists(model_dir):
            raise ValueError(
                f'{self.__class__.__name__} error in `save_model()`. '
                f'directory {model_dir} does not exists.'
            )

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = os.path.join(tmp,model_name.replace('.meteo',''))
            self.task.selector.vectorstore.save_local(tmp_path)

            with tarfile.open(model_path, mode="w:gz") as tar:
                tar.add(tmp_path, arcname=model_name)

        return self


    def load_model(
        self,
        model_path : str,
    ) -> Self:

        """
        Loads a model artifact to make new predictions.

        Parameters
        ----------

        model_path : str


        Returns
        -------

        self : Promptmeteo

        """

        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)

        if not model_name.endswith('.meteo'):
            raise ValueError(
                f'{self.__class__.__name__} error in `load_model()`. '
                f'model_path="{model_path}" has a bad model name extension. '
                f'Model name must end with `.meteo` (i.e. `./model.meteo`)'
            )

        if not os.path.exists(model_path):
            raise ValueError(
                f'{self.__class__.__name__} error in `load_model()`. '
                f'directory {model_dir} does not exists.'
            )

        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(model_path, "r:gz") as tar:
                tar.extractall(tmp)

            self.builder.build_selector_by_load(
                model_path         = os.path.join(tmp, model_name),
                selector_k         = self._selector_k,
                selector_algorithm = self._selector_algorithm
            )

        return self
