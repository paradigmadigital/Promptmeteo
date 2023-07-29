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

from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate


class BaseSelector():


    SELECTOR = None


    def __init__(
        self,
        embeddings      : Embeddings,
        k               : int
    ) -> None:

        self._k = k
        self._selector = None
        self._embeddings = embeddings

    @property
    def vectorstore(self):
        return self._selector.vectorstore


    @property
    def template(self) -> str:
        return self.run().format(__INPUT__='{__INPUT__}')


    def train(
        self,
        examples        : List[str],
        annotations     : List[str],
    ) -> None:

        examples = [{"ejemplo" : example, "respuesta" : annotation}
            for example, annotation in zip(examples,annotations)]

        self._selector = self.SELECTOR.from_examples(
            examples        = examples,
            embeddings      = self._embeddings,
            vectorstore_cls = FAISS,
            k               = self._k)


        return self


    def load_example_selector(
        self,
        model_path : str
    ):

        """
        """

        vectorstore = FAISS.load_local(model_path, self._embeddings)

        self._selector = self.SELECTOR(
            vectorstore=vectorstore,
            k=self._k)

        return self


    def run(self):

        """
        """

        if self._selector is None:
            raise Exception(
                f'`{self.__class__.__name__}` object has no vector store '
                f'created when executing `run()` method. You should call '
                f'method `load_example_selector()` `train()` befoto create '
                f'a vector store before.'
            )

        example_prompt = PromptTemplate(
            input_variables=["ejemplo", "respuesta"],
            template="Ejemplo: {ejemplo}\nRespuesta: {respuesta}",
        )

        suffix = "Ejemplo: {__INPUT__}\nRespuesta:"

        return FewShotPromptTemplate(
            example_selector=self._selector,
            example_prompt=example_prompt,
            suffix=suffix,
            input_variables=["__INPUT__"],
        )
