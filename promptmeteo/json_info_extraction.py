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


class JSONInfoExtraction(BaseUnsupervised):

    """
    Task for information extraction from text in JSON format.
    """

    @add_docstring_from(BaseUnsupervised.__init__)
    def __init__(
        self,
        language,
        fields_description: dict,
        **kwargs,
    ) -> None:
        """
        Example
        -------

        >>> from promptmeteo import JSONInfoExtraction

        >>> model = JSONInfoExtraction(
        >>>                  language="es",
        >>>                  fields_description = {
        >>>                         "topic":"Motivo de la llamada",
        >>>                         "sentiment":"Sentimiento del cliente",
        >>>                         "summary":"Resumen de la llamada",
        >>>                         "negative_tags":"Entidades y tópicos negativos en la llamada",
        >>>                         "positive_tags":"Entidades y tópicos positivos en la llamada"
        >>>                     },
        >>>                  model_name = "anthropic.claude-v2",
        >>>                  model_provider_name = "bedrock"
        >>>            )

        >>> model.predict(text)
        """

        kwargs["labels"] = None
        kwargs["language"] = language
        kwargs["fields_description"] = fields_description

        task_type = TaskTypes.JSON_INFO_EXTRACTION.value
        super(JSONInfoExtraction, self).__init__(**kwargs)

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
        
        # Setting the prompt description according to necessities
        prompt_detail = PromptTemplate.from_template(self.task.prompt.PROMPT_DETAIL)
        if len(set(prompt_detail.input_variables).intersection(set(["__FIELDS_DESCRIPTION__","__FIELDS__"]))) != 2:
            raise RuntimeError("Prompt file misses fields __FIELDS_DESCRIPTION__ or __FIELDS__")
        
        description_fields_str = "\n".join([f"{i+1}. {description} ({field})" 
                                            for i,field,description in 
                                            zip(range(len(fields_description)), 
                                                fields_description.keys(), fields_description.values())])
        prompt_detail = prompt_detail.format(__FIELDS__=",".join([i for i in fields_description.keys()]),
                        __FIELDS_DESCRIPTION__=description_fields_str)
        
        self.prompt_detail = prompt_detail
        self._builder.build_prompt(
            model_name=self.model_name,
            prompt_domain=self.prompt_domain,
            prompt_labels=self.prompt_labels,
            prompt_detail=self.prompt_detail,
        )
        self.builder.task.prompt.PROMPT_DETAIL = prompt_detail
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
        super(JSONInfoExtraction, self).train(examples=[""])

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
