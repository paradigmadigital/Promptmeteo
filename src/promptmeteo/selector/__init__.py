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
from typing import Dict

from langchain.embeddings.base import Embeddings


class SelectorAlgorithms(str, Enum):

    """
    """

    SELECTOR_1 = 'mmr'
    SELECTOR_2 = 'semantic_similarity'


class SelectorFactory():

    @staticmethod
    def factory_method(
        embeddings         : Embeddings,
        selector_k         : int,
        selector_algorithm : str,
    ):


        if selector_algorithm == SelectorAlgorithms.SELECTOR_1.value:
            from .marginal_relevance_selector import MMRSelector
            selector_cls = MMRSelector

        elif selector_algorithm == SelectorAlgorithms.SELECTOR_2.value:
            from .semantic_similarity_selector import SimSelector
            selector_cls = SimSelector

        else:
            raise ValueError(
               f'`SelectorFactory` error in `factory_method()` . '
               f'{selector_algorithm} is not in the list of supported '
               f'providers: {[i.value for i in SelectorAlgorithms]}')

        return selector_cls(
            embeddings=embeddings,
            k=selector_k
        )
