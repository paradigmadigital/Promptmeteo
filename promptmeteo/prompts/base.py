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

from abc import ABC
from string import Formatter
from typing import List

import yaml
from langchain.prompts import PromptTemplate


class BasePrompt(ABC):
    """
    Prompt class interface.
    """

    PROMPT_EXAMPLE = """
            TEMPLATE:
                "Here you explain the task.
                    {__PROMPT_DOMAIN__}
                    {__PROMPT_DETAIL__}

                    {__SHOT_EXAMPLES__}

                    {__PROMPT_SAMPLE__}
                    {__CHAIN_THOUGHT__}
                    {__ANSWER_FORMAT__}"

            PROMPT_SAMPLE:
                "Here you describe the task."

            SHOT_EXAMPLES:
                "Here you set the {__EXAMPLES__} to take into consideration."

            PROMPT_DOMAIN:
                "Here you explain the {__DOMAIN__} from the texts."

            PROMPT_LABELS:
                "Here you give the {__LABELS__} if required."

            PROMPT_DETAIL:
                "Here you can give some {__DETAIL__}"

            CHAIN_THOUGHT:
                "Explain your answer step by step."

            ANSWER_FORMAT:
                "Response just with the answer."
        """

    def __init__(
        self,
        prompt_domain: str = "",
        prompt_labels: str = "",
        prompt_detail: str = "",
    ) -> None:
        """
        Build a Prompt object string given a concrete especification.
        """

        self._prompt_domain = prompt_domain
        self._prompt_labels = prompt_labels
        self._prompt_detail = prompt_detail

    @property
    def domain(
        self,
    ) -> str:
        """Prompt Domain."""
        return self._prompt_domain

    @property
    def labels(
        self,
    ) -> List[str]:
        """Prompt Labels."""
        return [self._prompt_labels]

    @property
    def template(
        self,
    ) -> str:
        """Prompt Template."""
        return self.run().format(
            __SAMPLE__="{__SAMPLE__}", __EXAMPLES__="{__EXAMPLES__}"
        )

    @classmethod
    def read_prompt(
        cls,
        prompt_text: str,
    ) -> None:
        """
        Reads a Promptmeteo prompt string to build the Task Prompt. Promptmeteo
        prompts are expected to follow the following template:


        Parameters
        ----------

        prompt_text : str


        Returns
        -------

        Self

        """

        try:
            prompt = yaml.load(prompt_text, Loader=yaml.FullLoader)

        except Exception as error:
            raise ValueError(
                f"`{cls.__name__}` error in function `read_prompt()`. "
                f"The expected string input should be like:\n\n"
                f"{cls.PROMPT_EXAMPLE}\n\n{error}"
            ) from error

        try:
            cls.TEMPLATE = prompt["TEMPLATE"]
            cls.PROMPT_DOMAIN = prompt.get("PROMPT_DOMAIN", "")
            cls.PROMPT_LABELS = prompt.get("PROMPT_LABELS", "")
            cls.PROMPT_DETAIL = prompt.get("PROMPT_DETAIL", "")
            cls.PROMPT_SAMPLE = prompt.get("PROMPT_SAMPLE", "")
            cls.ANSWER_FORMAT = prompt.get("ANSWER_FORMAT", "")
            cls.CHAIN_THOUGHT = prompt.get("CHAIN_THOUGHT", "")
            cls.SHOT_EXAMPLES = prompt.get("SHOT_EXAMPLES", "")

            cls.PROMPT_EXAMPLE = prompt_text

        except Exception as error:
            raise ValueError(
                f"`{cls.__name__}` error `read_prompt()`. The expected keys "
                f"are {yaml.load(cls.PROMPT_EXAMPLE, Loader=yaml.FullLoader)}"
            ) from error

    def run(
        self,
    ) -> PromptTemplate:
        """
        Returns the prompt template for the current task.
        """

        prompt_variables = dict(
            __PROMPT_SAMPLE__=(
                self.PROMPT_SAMPLE if hasattr(self, "PROMPT_SAMPLE") else ""
            ),
            __PROMPT_LABELS__="",
            __PROMPT_DOMAIN__="",
            __PROMPT_DETAIL__="",
            __SHOT_EXAMPLES__=(
                self.SHOT_EXAMPLES if hasattr(self, "SHOT_EXAMPLES") else ""
            ),
            __ANSWER_FORMAT__=(
                self.ANSWER_FORMAT if hasattr(self, "ANSWER_FORMAT") else ""
            ),
            __CHAIN_THOUGHT__=(
                self.CHAIN_THOUGHT if hasattr(self, "CHAIN_THOUGHT") else ""
            ),
        )

        # Labels
        prompt_labels = (
            ", ".join(self._prompt_labels)
            if isinstance(self._prompt_labels, list)
            else self._prompt_detail
        )
        prompt_variables["__PROMPT_LABELS__"] = (
            self.PROMPT_LABELS.format(__LABELS__=prompt_labels)
            if self._prompt_labels
            else ""
        )

        # Domain
        prompt_variables["__PROMPT_DOMAIN__"] = self.PROMPT_DOMAIN.format(
            __DOMAIN__=self._prompt_domain
        )

        # Detail
        prompt_detail = (
            "\n - ".join([""] + self._prompt_detail)
            if isinstance(self._prompt_detail, list)
            else self._prompt_detail
        )
        prompt_variables["__PROMPT_DETAIL__"] = self.PROMPT_DETAIL.format(
            __DETAIL__=prompt_detail
        )

        return PromptTemplate.from_template(
            PromptTemplate.from_template(self.TEMPLATE).format(
                **{
                    k: v
                    for (k, v) in prompt_variables.items()
                    if k in [i[1] for i in Formatter().parse(self.TEMPLATE)]
                }
            )
        )
