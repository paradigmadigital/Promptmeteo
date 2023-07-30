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


class ClassificationParser():

    def __init__(
        self,
        prompt_labels               : List[str],
        prompt_labels_separator     : str = ',',
        prompt_chain_of_thoughts    : bool= False
    ) -> None:

        self._labels = prompt_labels
        self._labels_separator = prompt_labels_separator
        self._chain_of_thoughts = prompt_chain_of_thoughts


    def run(
        self,
        text : str
    ) -> str:

        text = text.lower()

        if not self._chain_of_thoughts:
            result = [word for word in text.split(self._labels_separator)
                if word in self._labels]

        if self._chain_of_thoughts:
            result = [label for label in self._labels if label in text]

        return result
