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
import yaml
from copy import deepcopy
from typing import List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .base import BaseSupervised
from .tasks import TaskTypes, TaskBuilder
from .tools import add_docstring_from


class APIGenerator(BaseSupervised):

    """
    API Generator Task.
    """

    @add_docstring_from(BaseSupervised.__init__)
    def __init__(
        self,
        language,
        n_samples: int,
        api_version: str,
        api_protocol: str,
        api_style_instructions: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Example
        -------

        >>> from promptmeteo import APIGenerator

        >>> model = APIGenerator(
        >>>     language='en',
        >>>     n_samples=5,
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

        >>> model.predict("API for managing account access")
        """

        if api_protocol not in ["REST"]:
            raise ValueError(
                f"{self.__class__.__name__} error in init function. "
                f"Available values for argument `api_protocol` are "
                f"{['gRPC', 'REST']}."
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
            kwargs["prompt_domain"] = f"OpenAPI {api_version} REST API YAML"

        kwargs["labels"] = None
        kwargs["language"] = language
        kwargs["selector_k"] = n_samples
        kwargs["n_samples"] = n_samples
        kwargs["api_version"] = api_version
        kwargs["api_protocol"] = api_protocol
        kwargs["api_style_instructions"] = api_style_instructions

        task_type = TaskTypes.API_GENERATION.value
        super(APIGenerator, self).__init__(**kwargs)

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

    @add_docstring_from(BaseSupervised.train)
    def train(
        self,
        api_descriptions: List[str],
        api_codes: List[str],
    ) -> Self:
        """
        Train the APIGenerator from a list of APIs and its descriptions.

        Parameters
        ----------

        api_descriptions : List[str]

        api_codes : List[str]


        Returns
        -------

        self

        """

        _api_descriptions = [
            description.split("\n")[0] for description in api_descriptions
        ]

        _api_codes = deepcopy(api_codes)
        for idx, api in enumerate(_api_codes):
            api = yaml.safe_load(api)
            api = self._prune_dict(api, max_depth=3)
            # api = self._sort_dict(api)
            api = yaml.dump(api, default_flow_style=False, sort_keys=False)

            _api_codes[idx] = "```\n" + api + "```"

        super(APIGenerator, self).train(
            examples=_api_descriptions,
            annotations=_api_codes,
        )

        return self

    def _prune_dict(
        self,
        old_dict: dict,
        max_depth: int = 3,
        current_depth: int = 1,
    ):
        """
        Prune a dictionary to a max_depth
        """

        new_dict = {}
        for key, val in old_dict.items():
            if current_depth < max_depth:
                if isinstance(val, dict):
                    new_val = self._prune_dict(
                        val, max_depth, current_depth + 1
                    )
                else:
                    new_val = val
                new_dict[key] = new_val
            elif current_depth == max_depth and isinstance(val, dict):
                new_dict[key] = [k for k in val.keys()]
            else:
                new_dict[key] = val

        new_dict = old_dict if max_depth == -1 else new_dict

        return new_dict

    def _sort_dict(
        self,
        old_dict: dict,
    ):
        """
        Sort the API dictionary in a concrete order for REST APIs
        """

        new_dict = {
            key: old_dict[key]
            for key in [
                "openapi",
                "info",
                "security",
                "servers",
                "tags",
                "paths",
                "components",
            ]
            if key in old_dict
        }

        new_dict.update(old_dict)

        return old_dict
