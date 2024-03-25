# %%
# !/usr/bin/python3
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
import tarfile
import tempfile
import json
import os

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .base import BaseUnsupervised
from .tasks import TaskTypes
from .tools import add_docstring_from


class Summarizer(BaseUnsupervised):
    """
    Class for text summarization

    This class represents a model for text summarization.

    Example
    -------
    >>> from promptmeteo import Summarizer
    >>> model = Summarizer(
    ...     language="es",
    ...     prompt_domain="A partir del siguiente texto:",
    ...     model_name="anthropic.claude-v2",
    ...     model_provider_name="bedrock"
    ... )

    >>> model.predict([text])
    """

    TASK_TYPE = TaskTypes.SUMMARIZATION.value

    @add_docstring_from(BaseUnsupervised.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize the Summarizer model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.
        """

        super(Summarizer, self).__init__(**kwargs)

    @add_docstring_from(BaseUnsupervised.train)
    def train(
        self,
    ) -> Self:
        """
        Train the Summarizer model.

        Returns
        -------
        self : Summarizer
            The trained Summarizer model.
        """
        super(Summarizer, self).train(examples=[""])

        return self

    @classmethod
    @add_docstring_from(BaseUnsupervised.load_model)
    def load_model(
        cls,
        model_path: str,
    ) -> Self:
        """
        Loads a model artifact to make new predictions.

        Parameters
        ----------
        model_path : str
            The path to the saved model artifact.

        Returns
        -------
        self : Summarizer
            The loaded Summarizer model.
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

            with open(init_tmp_path) as f_init:
                params = json.load(f_init)

            self = cls(**params)

            self.builder.build_selector_by_load(
                model_path=os.path.join(tmp, model_name),
                selector_type=self.SELECTOR_TYPE,
                selector_k=self._selector_k,
                selector_algorithm=self._selector_algorithm,
            )

            self._is_trained = True

        return self
