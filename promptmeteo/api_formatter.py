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

from .base import BaseUnsupervised
from .tasks import TaskTypes, TaskBuilder
from .tools import add_docstring_from


class APIFormatter(BaseUnsupervised):

    """
    API Generator Task.
    """

    @add_docstring_from(BaseUnsupervised.__init__)
    def __init__(
        self,
        language,
        api_version: str,
        api_protocol: str,
        api_style_instructions: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Example
        -------

        >>> from promptmeteo import APIFormatter

        >>> model = APIFormatter(
        >>>     language='en',
        >>>     api_version = '3.0.3',
        >>>     api_protocol = 'REST',
        >>>     api_style_instructions = [
        >>>         'Use always camel case.',
        >>>         'Do not use acronyms.'
        >>>         ],
        >>>     model_provider_name='openai',
        >>>     model_name='gpt-3.5-turbo-16k',
        >>>     model_provider_token=model_token,
        >>>     )

        >>> model.predict(api)
        """

        if api_protocol not in ["REST"]:
            raise ValueError(
                f"{self.__class__.__name__} error in init function. "
                f"Available values for argument `api_protocol` are "
                f"{['REST']}."
            )

        if (
            not re.compile(r"\d{1}\.\d\.\d").fullmatch(api_version)
            and api_protocol == "REST"
        ):
            raise ValueError(
                f"{self.__class__.__name__} error in init function. "
                f"Variable `api version` should be a correct version"
                f"example: '3.0.0' for REST"
            )

        if api_style_instructions is not None:
            if not all([isinstance(i, str) for i in api_style_instructions]):
                raise ValueError(
                    f"{self.__class__.__name__} error in init function. "
                    f"`api_style_instructions` argument should be a list "
                    f"of strings."
                )

            kwargs["prompt_detail"] = api_style_instructions

        if api_protocol == "REST":
            kwargs["prompt_domain"] = f"OpenAPI {api_version} REST API YAML."

        kwargs["labels"] = None
        kwargs["language"] = language
        kwargs["api_version"] = api_version
        kwargs["api_protocol"] = api_protocol
        kwargs["api_style_instructions"] = api_style_instructions

        task_type = TaskTypes.API_CORRECTION.value
        super(APIFormatter, self).__init__(**kwargs)

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

        # Build prompt
        self._builder.build_prompt(
            model_name=self.model_name,
            prompt_domain=self.prompt_domain,
            prompt_labels=self.prompt_labels,
            prompt_detail=self.prompt_detail,
        )

        # Build parser
        self._builder.build_parser(
            prompt_labels=self.prompt_labels,
        )

    @add_docstring_from(BaseUnsupervised.train)
    def train(
        self,
        api_codes: List[str],
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

        _api_codes = deepcopy(api_codes)

        _api_codes = [
            yaml.load(api, Loader=yaml.FullLoader) for api in _api_codes
        ]

        self._entities = {
            ntt_key: yaml.dump(
                ntt_val,
                default_flow_style=False,
                sort_keys=False,
            )
            for api in _api_codes
            for ntt_key, ntt_val in api.get("components", {})
            .get("schemas", {})
            .items()
        }

        self._parameters = {
            param_key: yaml.dump(
                param_val,
                default_flow_style=False,
                sort_keys=False,
            )
            for api in _api_codes
            for param_key, param_val in api.get("components", {})
            .get("parameters", {})
            .items()
        }

        self._init_params["_parameters"] = self._parameters
        self._init_params["_entities"] = self._entities

        super(APIFormatter, self).train(examples=[""])

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
            self._entities = params["_entities"]
            self._parameters = params["_parameters"]

            self.builder.build_selector_by_load(
                model_path=os.path.join(tmp, model_name),
                selector_type=self.SELECTOR_TYPE,
                selector_k=self._selector_k,
                selector_algorithm=self._selector_algorithm,
            )

            self._is_trained = True

        return self

    @add_docstring_from(BaseUnsupervised.predict)
    def predict(self, api_codes: List[str], external_info: dict) -> List[str]:
        """
        Receibe a list of API codes and return a list with the corrected APIs.

        Parameters
        ----------

        api_codes : List[str]


        Returns
        -------

        List[str]

        """

        _api_codes = deepcopy(api_codes)
        _api_codes = super(APIFormatter, self).predict(examples=_api_codes)
        _api_codes = [self._replace(api) for api in _api_codes]
        _api_codes = [
            self._add_external_information(api, external_info)
            for api in _api_codes
        ]
        return _api_codes

    def _replace(
        self,
        api: str,
    ) -> str:
        """
        Receive an API yaml string and replace the entities and parameters
        from the training data.
        """

        # Replace entities
        api = yaml.load(api, Loader=yaml.FullLoader)
        for entity in api.get("components", {}).get("schemas", {}).keys():
            if entity in self._entities:
                api["components"]["schemas"][entity] = yaml.load(
                    self._entities[entity], Loader=yaml.FullLoader
                )

        # Replace parameters
        for parameter in api.get("components", {}).get("parameters", {}).keys():
            if parameter in self._parameters:
                api["components"]["parameters"][parameter] = yaml.load(
                    self._parameters[parameter], Loader=yaml.FullLoader
                )

        # Complete entities
        while True:
            api_str = yaml.dump(api, default_flow_style=False, sort_keys=False)
            entities = re.findall(r"#/components/schemas/(\w+)", api_str)
            if all([ntt in api["components"]["schemas"] for ntt in entities]):
                break

            for ntt in entities:
                default_ntt = yaml.dump(
                    {
                        "title": ntt.lower(),
                        "type": "object",
                        "description": "WARNING: object created by default",
                    },
                    default_flow_style=False,
                    sort_keys=False,
                )

                api["components"]["schemas"][ntt] = yaml.load(
                    self._entities.get(ntt, default_ntt),
                    Loader=yaml.FullLoader,
                )

        # Complete parameters
        while True:
            api_str = yaml.dump(api, default_flow_style=False, sort_keys=False)
            params = re.findall(r"#/components/parameters/(\w+)", api_str)
            if all([par in api["components"]["parameters"] for par in params]):
                break

            for par in params:
                default_par = yaml.dump(
                    {
                        "name": par.lower(),
                        "in": "",
                        "description": "WARNING: object created by default",
                        "required": "false",
                        "schema": {},
                        "example": "",
                    },
                    default_flow_style=False,
                    sort_keys=False,
                )

                api["components"]["parameters"][par] = yaml.load(
                    self._entities.get(par, default_par),
                    Loader=yaml.FullLoader,
                )

        api = yaml.dump(api, default_flow_style=False, sort_keys=False)

        return api

    @staticmethod
    def _add_external_information(api: str, replacements: dict):
        def replace_values(orig_dict, replace_dict):
            for k, v in replace_dict.items():
                if k in orig_dict:
                    if isinstance(orig_dict[k], dict):
                        replace_values(orig_dict[k], v)
                    else:
                        orig_dict[k] = v
                else:
                    orig_dict[k] = v

            return orig_dict

        api = yaml.load(api, Loader=yaml.FullLoader)
        api = yaml.dump(
            replace_values(api, replacements),
            default_flow_style=False,
            sort_keys=False,
        )

        return api
