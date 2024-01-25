#%%
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
import re
import tarfile
import tempfile
import json
import os

import yaml
from copy import deepcopy
from typing import List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from langchain.prompts import PromptTemplate

from .base import BaseUnsupervised
from .tasks import TaskTypes, TaskBuilder
from .tools import add_docstring_from
from .validations import version_validation


class Summarizer(BaseUnsupervised):

    """
    Class for text summarization
    """
    @add_docstring_from(BaseUnsupervised.__init__)
    def __init__(
        self,
        prompt_domain,
        language,
        **kwargs,
    ) -> None:
        """
        Example
        -------
        """

        kwargs["labels"] = None
        kwargs["language"] = language
        kwargs["prompt_domain"] = prompt_domain


        task_type = TaskTypes.SUMMARIZATION.value
        super(Summarizer, self).__init__(**kwargs)

        self._builder = TaskBuilder(
            language=self.language,
            task_type=task_type,
            verbose=self.verbose,
        )

        # Build model
        self._builder.build_model(
            model_name=self.model_name,
            model_provider_name=self.model_provider_name,
            model_provider_token=self.model_provider_token,
            model_params=self.model_params,
        )

        # Building prompt
        self._builder.build_prompt(
            model_name=self.model_name,
            prompt_domain=self.prompt_domain,
            prompt_labels=self.prompt_labels,
            prompt_detail=self.prompt_detail,
        )
        
        ##
        # Build parser
        self._builder.build_parser(
            prompt_labels=self.prompt_labels,
        )
        

        

    @add_docstring_from(BaseUnsupervised.train)
    def train(
        self,
    ) -> Self:
        """
        Train the APIFormatter to extract entities anda parameteres.

        Parameters
        ----------

        api_codes : List[str]


        Returns
        -------

        self

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

    # @add_docstring_from(BaseUnsupervised.predict)
    # def predict(self, api_codes: List[str], external_info: dict) -> List[str]:
    #     """
    #     Receibe a list of API codes and return a list with the corrected APIs.

    #     Parameters
    #     ----------

    #     api_codes : List[str]


    #     Returns
    #     -------

    #     List[str]

    #     """

    #     _api_codes = deepcopy(api_codes)
    #     _api_codes = super(JSONInfoExtractor, self).predict(examples=_api_codes)
    #     _api_codes = [self._replace(api) for api in _api_codes]
    #     _api_codes = [
    #         self._add_external_information(api, external_info)
    #         for api in _api_codes
    #     ]
    #     return _api_codes

#%%
# JSON_SUMMARIZATION = JSONInfoExtractor(language="es",
#                                  json_fields=["summary","sentiment","topic","keywords"],
#                                  fields_description={"summary":"",
#                                                      "sentiment":"",
#                                                      "topic":"",
#                                                      "keywords":""})
