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
import json
from abc import ABC
from inspect import signature
from typing import List
from typing import Dict
from typing import Optional

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .tasks import Task
from .tasks import TaskBuilder
from .tools import add_docstring_from


class Base(ABC):

    """
    'Sun is setting on the New Republic. It's time for the ResistencIA to rise'

                                         - Padme Amidala, mother of Leia -
    """
    SELECTOR_TYPE: str = ''

    def __init__(
        self,
        language: str,
        model_name: str,
        model_provider_name: str,
        model_provider_token: Optional[str] = None,
        model_params: Optional[Dict] = None,
        prompt_domain: Optional[str] = "",
        prompt_labels: List[str] = None,
        prompt_detail: Optional[str] = None,
        selector_k: int = 10,
        selector_algorithm: str = "relevance",
        verbose: bool = False,
        **kwargs
    ) -> None:
        """
        Prompmeteo is tool, powered by LLMs, which is able to solve NLP models
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
        """

        _init_params: dict = {}
        _local = locals()

        for param in signature(self.__class__).parameters:
            if 'kwargs' in param:
                continue
            _init_params[param] = _local.get(param, kwargs.get(param))

        for param in signature(Base).parameters:
            if 'kwargs' in param:
                continue
            _init_params[param] = _local.get(param, kwargs.get(param))

        self._init_params: dict = _init_params

        self.language: str = language
        self.model_name: str = model_name
        self.model_provider_name: str = model_provider_name
        self.model_provider_token: Optional[str] = model_provider_token
        self.model_params: Dict = model_params or {}
        self.prompt_domain: Optional[str] = prompt_domain
        self.prompt_labels: List[str] = prompt_labels or []
        self.prompt_detail: Optional[str] = prompt_detail
        self._selector_k: int = selector_k
        self._selector_algorithm: str = selector_algorithm
        self.verbose: bool = verbose

        self._builder = None
        self._is_trained = False

    @property
    def builder(
        self,
    ) -> TaskBuilder:
        """
        Get TaskBuilder.
        """
        return self._builder

    @property
    def task(
        self,
    ) -> Task:
        """
        Get Task.
        """
        return self._builder.task

    @property
    def is_trained(
        self,
    ) -> bool:
        """
        Check if model intance is trained.
        """
        return self._is_trained

    def predict(
        self,
        examples: List[str],
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

        if not isinstance(examples, list):
            raise ValueError(
                f"{self.__class__.__name__} error in function `predict()`. "
                f"Arguments `examples` and `annotations` are expected to be "
                f"of type `List[str]`. Instead they got: `{type(examples)}`"
            )

        if not all([isinstance(val, str) for val in examples]):
            raise ValueError(
                f"{self.__class__.__name__} error in function `predict()`. "
                f"Arguments `examples` are expected to be of type "
                f"  `List[str]`. Some values seem no to be of type `str`."
            )

        results = []
        for example in examples:
            results.append(self.task.run(example))

        return results

    def save_model(
        self,
        model_path: str,
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
                f"{self.__class__.__name__} error in `save_model()`. "
                f"You are trying to save a model that has non been trained. "
                f"Please, call `train()` function before"
            )

        if not model_name.endswith(".meteo"):
            raise ValueError(
                f"{self.__class__.__name__} error in `save_model()`. "
                f'model_path="{model_path}" has a bad model name extension. '
                f"Model name must end with `.meteo` (i.e. `./model.meteo`)"
            )

        if model_dir != "" and not os.path.exists(model_dir):
            raise ValueError(
                f"{self.__class__.__name__} error in `save_model()`. "
                f"directory {model_dir} does not exists."
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = os.path.join(tmp, model_name.replace(".meteo", ""))
            self.task.selector.vectorstore.save_local(tmp_path)

            init_tmp_path = f"{tmp_path}.init"
            with open(init_tmp_path, mode='w') as f:
                json.dump(self._init_params, f)

            with tarfile.open(model_path, mode="w:gz") as tar:
                tar.add(tmp_path, arcname=model_name)
                tar.add(init_tmp_path, arcname=os.path.basename(init_tmp_path))

        return self

    @classmethod
    def load_model(
        cls,
        model_path: str,
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

        if not model_name.endswith(".meteo"):
            raise ValueError(
                f"{cls.__name__} error in `load_model()`. "
                f'model_path="{model_path}" has a bad model name extension. '
                f"Model name must end with `.meteo` (i.e. `./model.meteo`)"
            )

        if not os.path.exists(model_path):
            raise ValueError(
                f"{cls.__name__} error in `load_model()`. "
                f"directory {model_dir} does not exists."
            )

        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(model_path, "r:gz") as tar:
                tar.extractall(tmp)

            init_tmp_path = os.path.join(tmp, f"{os.path.splitext(model_name)[0]}.init")
            with open(init_tmp_path) as f:
                self = cls(**json.load(f))

            self.builder.build_selector_by_load(
                model_path=os.path.join(tmp, model_name),
                selector_type=self.SELECTOR_TYPE,
                selector_k=self._selector_k,
                selector_algorithm=self._selector_algorithm,
            )

            self._is_trained = True

        return self


class BaseSupervised(Base):

    """
    Model Inferface for supervised training tasks.
    """

    SELECTOR_TYPE = "supervised"

    @add_docstring_from(Base.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        """

        super(BaseSupervised, self).__init__(**kwargs)

    def train(
        self,
        examples: List[str],
        annotations: List[str],
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

        if not isinstance(examples, list) or not isinstance(annotations, list):
            raise ValueError(
                f"Arguments `examples` and `annotations` are expected to be of "
                f"type `List[str]`. Instead they got: examples {type(examples)}"
                f" annotations {type(annotations)}."
            )

        if not all([isinstance(val, str) for val in examples]) or not all(
            [isinstance(val, str) for val in annotations]
        ):
            raise ValueError(
                f"{self.__class__.__name__} error in function `train()`. "
                f"Arguments `examples` and `annotations` are expected to be of "
                f"type `List[str]`. Some values seem no to be of type `str`."
            )

        if len(examples) != len(annotations):
            raise ValueError(
                f"{self.__class__.__name__} error in function `train()`. "
                f"Arguments `examples` and `annotations` are expected to have "
                f"the same length. examples=({len(examples)},) annotations= "
                f"({len(annotations)},)"
            )

        if self.prompt_labels:
            for idx, annotation in enumerate(annotations):
                if annotation not in self.prompt_labels:
                    raise ValueError(
                        f"{self.__class__.__name__} error in `train()`. "
                        f"`annotation value in item {idx}: `{annotation}`"
                        f"is not in the expected values: {self.prompt_labels}"
                    )

        self.builder.build_selector_by_train(
            examples=examples,
            annotations=annotations,
            selector_k=self._selector_k,
            selector_type=self.SELECTOR_TYPE,
            selector_algorithm=self._selector_algorithm,
        )

        self._is_trained = True

        return self


class BaseUnsupervised(Base):
    """
    Model Inferface for unsupervised training tasks.
    """

    SELECTOR_TYPE = "unsupervised"

    @add_docstring_from(Base.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        if "prompt_labels" in kwargs:
            raise ValueError(
                f"{self.__class__.__name__} can not be inicializated with the "
                f"argument `prompt_labels`."
            )

        super(BaseUnsupervised, self).__init__(**kwargs)

    def train(
        self,
        examples: List[str],
    ) -> Self:
        """
        Trains the model given examples and its annotations. The training
        process create a vector store with all the training texts in memory

        Parameters
        ----------

        examples : List[str]


        Returns
        -------

        self

        """

        if not isinstance(examples, list):
            raise ValueError(
                f"{self.__class__.__name__} error in function `train()`. "
                f"Arguments `examples` are expected to be of type `List[str]`."
                f"Instead they got: examples {type(examples)}"
            )

        if not all([isinstance(val, str) for val in examples]):
            raise ValueError(
                f"{self.__class__.__name__} error in function `train()`. "
                f"Arguments `examples` are expected to be of type `List[str]`."
                f"Some values seem no to be of type `str`."
            )

        self.builder.build_selector_by_train(
            examples=examples,
            annotations=None,
            selector_k=self._selector_k,
            selector_type=self.SELECTOR_TYPE,
            selector_algorithm=self._selector_algorithm,
        )

        self._is_trained = True

        return self
