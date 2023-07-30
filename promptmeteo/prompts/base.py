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
import yaml
from typing import List

from langchain import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate


def _format(
    text : str
    ) -> str:

    """
    Example
    -------

    >>> texto = '''
    >>>     Este es un texto
    >>>     identado siguiendo
    >>>     el estilo de python
    >>>     '''
    >>>
    >>> print(text)

    '    \nEste es un texto    \nidentado siguiendo    \n estilo de python    '

    >>> print(format_strin(text))

    'Este es un texto identado siguiendo estilo de python'
    """

    return text.replace(' '*4,''
        ).replace('\n\n','|'
        ).replace('\n',' '
        ).replace('|','\n\n'
        ).replace('\n ','\n')


class BasePrompt():


    PROMPT_EXAMPLE = _format("""
        TEMPLATE:
            'Explain the main goal of the task:
            {__LABELS__}.

            {__TASK_INFO__}

            {__ANSWER_FORMAT__}

            {__CHAIN_OF_THOUGHTS__}'

        LABELS:
            ['label_1','label_2','label_3']

        TASK_INFO:
            'Explain the task in detail and the language dommain.'

        ANSWER_FORMAT:
            'Explain the format of the output'

        CHAIN_OF_THOUGHTS:
            'Explain the reasoning process to achieve the goal'
    """)


    def __init__(
        self,
        prompt_labels            : str = '',
        prompt_task_info         : str = '',
        prompt_answer_format     : str = '',
        prompt_chain_of_thoughts : str = '',
    ) -> None:

        """
        Build a Prompt object string given a concrete especification

        Parameters
        ----------

        prompt_labels : List[str]
            List with all the labels.

        prompt_task_info : str
            Text explaining the classification task and the label meanings.

        prompt_answer_format : str
            Text explaining the output format.

        prompt_chain_of_thoughts : str
            Text explaining the reasoning task step by step.

        Returns
        --------

        prompt : Prompt
            Text with the reasulting prompt.

        Example
        -------
        ```
        >>> from promptmeteo.prompts.sp import BasePrompt

        >>> BasePrompt(
        >>>     prompt_labels            = BasePrompt.PROMPT_LABELS,
        >>>     prompt_task_info         = BasePrompt.PROMPT_TASK_INFO,
        >>>     prompt_answer_format     = BasePrompt.PROMPT_ANSWER_FORMAT,
        >>>     prompt_chain_of_thoughts = BasePrompt.PROMPT_CHAIN_OF_THOUGHTS,
        >>> ).build()

        Clasifica el siguiente texto en una de las siguientes clases:

        positivo, negativo, neutro.

        Asume que estamos usando estas categorías con su significado subjetivo
        habitual: algo positivo puede describirse como bueno, de buena calidad,
        deseable, útil y satisfactorio; algo negativo puede describirse como
        malo, de mala calidad, indeseable, inútil o insatisfactorio; neutro es
        la clase que asignaremos a todo lo que no sea positivo o negativo.

        En tu respuesta incluye sólo el nombre de la clase, como una única
        palabra (positivo, negativo, neutro), en minúscula, sin puntuación, y
        sin añadir ninguna otra afirmación o palabra.

        Por favor argumenta tu respuesta paso a paso, explica por qué crees
        que está justificada tu elección final, y por favor asegúrate de que
        acabas tu explicación con el nombre de la clase que has escogido como
        la correcta, en minúscula y sin puntuación.

        Éste es el texto que debes clasificar: "{__INPUT__}"
        ```
        """

        self._labels = prompt_labels

        prompt_labels = PromptTemplate.from_template(_format(
            ', '.join(prompt_labels)))

        prompt_task_info = PromptTemplate.from_template(_format(
            prompt_task_info))

        prompt_answer_format = PromptTemplate.from_template(_format(
            prompt_answer_format))

        prompt_chain_of_thoughts = PromptTemplate.from_template(_format(
            prompt_chain_of_thoughts))

        self._prompt = PipelinePromptTemplate(
            final_prompt=PromptTemplate.from_template(
                BasePrompt.PROMPT_TEMPLATE),
            pipeline_prompts=[
                ('__LABELS__', prompt_labels),
                ('__TASK_INFO__', prompt_task_info),
                ('__ANSWER_FORMAT__', prompt_answer_format),
                ('__CHAIN_OF_THOUGHTS__', prompt_chain_of_thoughts)])


    @property
    def labels(self) -> List[str]:
        return self._labels


    @property
    def template(self) -> str:
        return self._prompt.format()


    @classmethod
    def read_prompt_file(cls, prompt_text : str) -> None:

        """
        """

        try:

            prompt = yaml.load(
                prompt_text,
                Loader=yaml.FullLoader)

            BasePrompt.PROMPT_LABELS = \
                prompt['LABELS']

            BasePrompt.PROMPT_TEMPLATE = \
                prompt['TEMPLATE']

            BasePrompt.PROMPT_TASK_INFO = \
                prompt['TASK_INFO']

            BasePrompt.PROMPT_ANSWER_FORMAT = \
                prompt['ANSWER_FORMAT']

            BasePrompt.PROMPT_CHAIN_OF_THOUGHTS = \
                prompt['CHAIN_OF_THOUGHTS']

            BasePrompt.PROMPT_EXAMPLE = prompt_text

        except Exception as error:
            raise Exception(
                f'`cls.__name__ class error. `read_prompt_file` is trying '
                f'to read prompt with a wrong prompt template format. '
                f'The expected string input should be like this: \n\n'
                f'{BasePrompt.PROMPT_EXAMPLE}'
                ) from error


    def run(self):
        return self._prompt
