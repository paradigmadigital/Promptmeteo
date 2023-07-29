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

from enum import Enum
from typing import List


class ParserTypes(str, Enum):

    """
    """

    PARSER_1 = "classification"
    PARSER_2 = "ner"

class ParserFactory():

    """
    The ModelFactory class is used to create a BaseModel object from the given
    AutoLabelConfig configuration.
    """

    @staticmethod
    def factory_method(
        parser_type                 : str,
        prompt_labels               : List[str],
        prompt_labels_separator     : str = ',',
        prompt_chain_of_thoughts    : bool= False
    ):
        if parser_type == ParserTypes.PARSER_1.value:
            from .classification_parser import ClassificationParser
            parser_cls = ClassificationParser

        elif parser_type == ParserTypes.PARSER_2.value:
            from .classification_parser import ClassificationParser
            parser_cls = ClassificationParser

        else:
            raise Exception(
                f"{parser_type} is not in the list of supported providers: "
                f"{[i.value for i in ParserTypes]}"
                )

        return parser_cls(
            prompt_labels=prompt_labels,
            prompt_labels_separator=prompt_labels_separator,
            prompt_chain_of_thoughts=prompt_chain_of_thoughts
        )
