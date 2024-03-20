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

import yaml
from copy import deepcopy
from typing import List, Optional

from .constants import REST_PROTOCOL

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .base import BaseSupervised
from .tasks import TaskTypes
from .tools import add_docstring_from
from .validations import version_validation


class APIGenerator(BaseSupervised):
    """
    API Generator Task.

    This class initializes the APIGenerator Task to create APIs based on descriptions.

    Parameters
    ----------
    language : str
        Language for the API descriptions.
    model_name : str
        Name of the model to be used for API generation.
    model_provider_name : str
        Name of the model provider.
    api_version : str
        Version of the API.
    api_protocol : str
        Protocol of the API.
    api_style_instructions : Optional[List[str]]
        Instructions for API style.
    **kwargs : dict
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If `api_protocol` is not in the allowed protocols or if `api_version`
        is not in the correct format.

    Example
    -------
    >>> from promptmeteo import APIGenerator
    >>> model = APIGenerator(
    ...     language='en',
    ...     selector_k=5,
    ...     api_version='3.0.3',
    ...     api_protocol='REST',
    ...     api_style_instructions=[
    ...         'Use always camel case.',
    ...         'Do not use acronyms.'
    ...     ],
    ...     model_provider_name='openai',
    ...     model_name='gpt-3.5-turbo-16k',
    ...     model_provider_token=model_token,
    ... )

    >>> model.train(api_description, api_code)

    >>> model.predict("API for managing account access")
    """

    TASK_TYPE = TaskTypes.API_GENERATION.value
    ALLOWED_PROTOCOLS = [REST_PROTOCOL]

    @add_docstring_from(BaseSupervised.__init__)
    def __init__(
        self,
        language: str,
        model_name: str,
        model_provider_name: str,
        api_version: str,
        api_protocol: str,
        api_style_instructions: Optional[List[str]],
        **kwargs,
    ) -> None:

        kwargs.setdefault("selector_k", 5)

        if api_protocol not in self.ALLOWED_PROTOCOLS:
            raise ValueError(
                f"{self.__class__.__name__} error in init function. "
                f"Available values for argument `api_protocol` are "
                f"{self.ALLOWED_PROTOCOLS}."
            )

        if version_validation(api_version, api_protocol):
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

        kwargs.setdefault("prompt_detail", api_style_instructions)

        if api_protocol == REST_PROTOCOL:
            kwargs.setdefault(
                "prompt_domain", f"OpenAPI {api_version} REST API YAML."
            )

        kwargs["language"] = language
        kwargs["model_name"] = model_name
        kwargs["model_provider_name"] = model_provider_name

        super(APIGenerator, self).__init__(**kwargs)
        self._init_params.update(
            {
                "api_version": api_version,
                "api_protocol": api_protocol,
                "api_style_instructions": api_style_instructions,
            }
        )

    @add_docstring_from(BaseSupervised.train)
    def train(
        self,
        api_descriptions: List[str],
        api_codes: List[str],
    ) -> Self:
        """
        Train the APIGenerator from a list of APIs descriptions and contracts.

        Parameters
        ----------
        api_descriptions : List[str]
            List of API descriptions.
        api_codes : List[str]
            List of API codes - contracts.

        Returns
        -------
        APIGenerator
            Returns the trained APIGenerator object.
        """

        _api_descriptions = [
            description.split("\n")[0] for description in api_descriptions
        ]

        _api_codes = deepcopy(api_codes)
        for idx, api in enumerate(_api_codes):
            api = yaml.safe_load(api)
            api = self._prune_dict(api, max_depth=3)
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
        Prune a dictionary to a specified maximum depth.

        Parameters
        ----------
        old_dict : dict
            Dictionary to be pruned.
        max_depth : int, optional
            Maximum depth to which the dictionary should be pruned, by default 3.
        current_depth : int, optional
            Current depth of recursion, by default 1.

        Returns
        -------
        dict
            Pruned dictionary.
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
