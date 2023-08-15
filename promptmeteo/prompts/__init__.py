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
from enum import Enum
from typing import List

from .base import BasePrompt


module_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))


def get_files_taxonomy(sep: str = "_"):
    """
    Convert a list of prompt files with the naming convetion of
    `{model_name}_{language}_{task}.prompt into a dictorionary version.
    Given the following folder structure:

        ./prompts/
          ├── __init__.py
          ├── base.py
          ├── google-flan-t5-small_es_classification.prompt
          ├── google-flan-t5-small_es_code-generation.prompt
          ├── google-flan-t5-small_es_ner.prompt
          ├── text-davinci-003_es_classification.prompt
          ├── text-davinci-003_es_code-generation.prompt
          └── text-davinci-003_es_ner.prompt

    The function returns:

        {'google-flan-t5-small': {
             'es': {
                 'classification': {},
                 'ner': {}
                 }
              },
         'text-davinci-003': {
             'es': {
                 'classification': {},
                 'code-generation': {},
                 'ner': {}
                 }
              }
        }

    """

    try:
        prompt_files = [
            os.path.join(path, name.replace(".prompt", "")).replace(
                module_dir + "/", ""
            )
            for path, _, files in os.walk(module_dir)
            for name in files
            if name.endswith(".prompt")
        ]

        taxonomy = {}
        for prompt_file in prompt_files:
            tmp = taxonomy
            for field in prompt_file.split(sep):
                tmp = tmp.setdefault(field, {})

    except Exception as error:
        raise RuntimeError(
            "Problems with the prompts file structure in directory "
            "promptmeteo/prompts. The expected naming for the prompts "
            "is `<model_name>_<language>_<task>.prompt`."
        ) from error

    return taxonomy


class PromptFactory:

    """
    Factory of Prompts
    """

    @classmethod
    def factory_method(
        cls,
        language: str,
        task_type: str,
        model_name: str,
        prompt_domain: str,
        prompt_labels: List[str],
        prompt_detail: str,
    ):
        """
        Returns and instance of a BasePrompt object depending on the
        `task_type`.
        """

        _model_name = model_name.replace("/", "-")

        taxonomy = get_files_taxonomy()

        if _model_name not in taxonomy:
            raise ValueError(
                f"`{cls.__name__}` class in function `factory_method()`. "
                f"{model_name} has not a prompt file created. Available model "
                f"prompts are: {list(taxonomy)}"
            )

        if language not in taxonomy[_model_name]:
            raise ValueError(
                f"`{cls.__name__}` class in function `factory_method()`. "
                f"{model_name} has not a prompt file created for the language "
                f"{language}. Available languages for {model_name}: "
                f"{list(taxonomy[model_name])}"
            )

        if task_type not in taxonomy[_model_name][language]:
            raise ValueError(
                f"`{cls.__name__}` class in function `factory_method()`. "
                f"{model_name}  in {language} has not a prompt file created"
                f"for the task {task_type}. Available tasks are: "
                f"{list(taxonomy[model_name][language])}"
            )

        prompt_cls = cls.build_class(language, task_type, _model_name)

        return prompt_cls(
            prompt_domain=prompt_domain,
            prompt_labels=prompt_labels,
            prompt_detail=prompt_detail,
        )

    @staticmethod
    def build_class(
        language: str,
        task_type: str,
        _model_name: str,
    ) -> BasePrompt:
        """
        Creates a class dinamically that inherits from `BasePrompt` given
        arguments configuracion. This new class has included its prompt from
        the .prompt file.

        Example
        -------

        >>> PromptFactory.build_class(
        >>>     language='es',
        >>>     task_type='ner',
        >>>     model_name='text-davinci-003')
        >>> )

        <class '__main__.Textdavinci003SpNer'>
        """

        prompt_class_name = "".join(
            [
                _model_name.replace("-", "").capitalize(),
                language.capitalize(),
                task_type.replace("-", "").capitalize(),
            ]
        )

        prompt_cls = type(
            prompt_class_name, (BasePrompt,), {"__init__": BasePrompt.__init__}
        )

        file_name = f"{_model_name}_{language}_{task_type}.prompt"
        with open(os.path.join(module_dir, file_name), encoding="utf-8") as fin:
            prompt_cls.read_prompt(fin.read())

        return prompt_cls
