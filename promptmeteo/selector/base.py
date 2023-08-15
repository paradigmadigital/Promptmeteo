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

from typing import List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate


class BaseSelector:

    """
    Base Selector Interface
    """

    SELECTOR = None

    def __init__(
        self,
        language: str,
        embeddings: Embeddings,
        k: int,
    ) -> None:
        self._k = k
        self._selector = None
        self._language = language
        self._embeddings = embeddings

    @property
    def vectorstore(self):
        """Selector Vectorstore."""
        return self._selector.vectorstore

    @property
    def template(
        self,
    ) -> str:
        """Selector Template"""
        return self.run().format(__INPUT__="{__INPUT__}")

    def train(
        self,
        examples: List[str],
        annotations: List[str],
    ) -> Self:
        """
        Creates the vectorstor with the training samples.
        """

        examples = [
            {"__INPUT__": example, "__OUTPUT__": annotation}
            for example, annotation in zip(examples, annotations)
        ]

        self._selector = self.SELECTOR.from_examples(
            examples=examples,
            embeddings=self._embeddings,
            vectorstore_cls=FAISS,
            k=self._k,
        )

        return self

    def load_example_selector(
        self,
        model_path: str,
    ) -> Self:
        """
        Load a vectorstore database from a disk file
        """

        vectorstore = FAISS.load_local(model_path, self._embeddings)

        self._selector = self.SELECTOR(vectorstore=vectorstore, k=self._k)

        return self

    def run(
        self,
    ) -> FewShotPromptTemplate:
        """
        Creates the FewShotPromptTemplate from the samples of the vectorstore.
        """

        if self._selector is None:
            raise RuntimeError(
                f"`{self.__class__.__name__}` object has no vector store "
                f"created when executing `run()` method. You should call "
                f"method `load_example_selector()` `train()` befoto create "
                f"a vector store before."
            )

        template = "{__INPUT__}\n{__OUTPUT__}"

        if self._language == "es":
            template = "EJEMPLO: {__INPUT__}\nRESPUESTA: {__OUTPUT__}"

        if self._language == "en":
            template = "INPUT: {__INPUT__}\nOUTPUT: {__OUTPUT__}"

        example_prompt = PromptTemplate(
            input_variables=["__INPUT__", "__OUTPUT__"],
            template=template,
        )

        return FewShotPromptTemplate(
            example_selector=self._selector,
            example_prompt=example_prompt,
            suffix=template.format(__INPUT__="{__INPUT__}", __OUTPUT__=""),
            input_variables=["__INPUT__"],
        )
