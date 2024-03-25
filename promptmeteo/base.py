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
from typing import (
    List,
    Any,
    Dict,
    Optional,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .tasks import Task
from .tasks import TaskBuilder
from .tools import add_docstring_from
from .selector.base import SelectorAlgorithms


class Base(ABC):
    """
    Promptmeteo is a tool powered by LLMs, capable of solving NLP tasks such as
    text classification and Named Entity Recognition. Its interface resembles
    that of a conventional ML model, making it easy to integrate into MLOps
    pipelines.

    Notes
    ----------
        'Sun is setting on the New Republic. It's time for the ResistencIA to rise'
            Padmé Amidala, mother of Leia
    """

    TASK_TYPE: str = ""
    SELECTOR_TYPE: str = ""

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
        **kwargs,
    ) -> None:
        """
        Initialize the Promptmeteo model.

        Raises
        ------
        ValueError
            If the selector algorithm is invalid.
        """

        self._init_params: Dict[str, Any] = {
            "language": language,
            "model_name": model_name,
            "model_provider_name": model_provider_name,
            "model_provider_token": model_provider_token,
            "model_params": model_params,
            "prompt_domain": prompt_domain,
            "prompt_labels": prompt_labels,
            "prompt_detail": prompt_detail,
            "selector_k": selector_k,
            "selector_algorithm": selector_algorithm,
            "verbose": verbose,
        }
        self._init_params.update(kwargs)

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
        if (
            self._selector_algorithm
            == SelectorAlgorithms.SIMILARITY_CLASS_BALANCED.value
        ) and (self.__class__.__name__ != "DocumentClassifier"):
            raise ValueError(
                f"{self.__class__.__name__} error in function `__init__`. "
                f"Selector algorithm {self._selector_algorithm} "
                f"is only valid for DocumentClassifier models"
            )
        self.verbose: bool = verbose

        self._builder = None
        self._is_trained = False

    @property
    def init_params(self):
        """
        Get the initialization parameters of the model.
        """
        return self._init_params

    @property
    def builder(
        self,
    ) -> TaskBuilder:
        """
        Get the TaskBuilder instance for the model.
        """
        if self._builder is None:
            self._builder = self.create_builder()
        return self._builder

    def create_builder(self) -> TaskBuilder:
        """
        Create a TaskBuilder instance for the model.
        """
        builder = TaskBuilder(
            language=self.language,
            task_type=self.TASK_TYPE,
            verbose=self.verbose,
        )

        # Build model
        builder.build_model(
            model_name=self.model_name,
            model_provider_name=self.model_provider_name,
            model_provider_token=self.model_provider_token,
            model_params=self.model_params,
        )

        # Build prompt
        builder.build_prompt(
            model_name=self.model_name,
            prompt_domain=self.prompt_domain,
            prompt_labels=self.prompt_labels,
            prompt_detail=self.prompt_detail,
        )

        # Build parser
        builder.build_parser(
            prompt_labels=self.prompt_labels,
        )
        return builder

    @property
    def task(
        self,
    ) -> Task:
        """
        Get the Task instance for the model.
        """
        return self.builder.task

    @property
    def is_trained(
        self,
    ) -> bool:
        """
        Check if the model is trained.
        """
        return self._is_trained

    def predict(
        self,
        examples: List[str],
    ) -> List[str]:
        """
        Predict over new text samples.

        Parameters
        ----------
        examples : List[str]
            List of text samples to predict.

        Returns
        -------
        List[str]
            List of predictions.
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
        Save the trained model to disk.

        Parameters
        ----------
        model_path : str
            Path where the model will be saved.

        Returns
        -------
        Base
            Instance of the saved model.
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
            with open(init_tmp_path, mode="w") as f:
                json.dump(self.init_params, f)

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
        Load a saved model from disk.

        Parameters
        ----------
        model_path : str
            Path from where the model will be loaded.

        Returns
        -------
        Base
            Loaded model instance.
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

            init_tmp_path = os.path.join(
                tmp, f"{os.path.splitext(model_name)[0]}.init"
            )
            with open(init_tmp_path) as f:
                self = cls(**json.load(f))

            self._load_builder(model_path=os.path.join(tmp, model_name))

            self._is_trained = True

        return self

    def _load_builder(self, **kwargs) -> TaskBuilder:
        kwargs.setdefault("selector_type", self.SELECTOR_TYPE)
        kwargs.setdefault("selector_k", self._selector_k)
        kwargs.setdefault("selector_algorithm", self._selector_algorithm)
        kwargs.setdefault("input_keys", ["__INPUT__"]),
        kwargs.setdefault("class_list", self.prompt_labels)
        kwargs.setdefault("class_key", "__OUTPUT__")

        return self.builder.build_selector_by_load(**kwargs)


class BaseSupervised(Base):
    """
    Base class for supervised training tasks.
    """

    SELECTOR_TYPE = "supervised"

    @add_docstring_from(Base.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize the supervised training model.
        """

        super(BaseSupervised, self).__init__(**kwargs)

    def train(
        self,
        examples: List[str],
        annotations: List[str],
    ) -> Self:
        """
        Trains the model given use cases and notes on behaviour. Check the
        parameters and task behaviour in each specific model training docstring.
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

        examples = [
            example.replace("{", "{{").replace("}", "}}")
            for example in examples
        ]

        annotations = [
            annotation.replace("{", "{{").replace("}", "}}")
            for annotation in annotations
        ]

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
    Base class for unsupervised training tasks.
    """

    SELECTOR_TYPE = "unsupervised"

    @add_docstring_from(Base.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize the unsupervised training model.
        """
        if kwargs.get("prompt_labels", None):
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
        Trains the model given use cases and notes on behaviour. Check the
        parameters and task behaviour in each specific model training docstring.
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
