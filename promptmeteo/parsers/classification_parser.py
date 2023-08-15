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

from .base import BaseParser


class ClassificationParser(BaseParser):

    """
    Parser for the classification task.
    """

    def run(
        self,
        text: str,
    ) -> List[str]:
        """
        Given a response string from an LLM, returns the response expected for
        the task.
        """

        text = self._preprocess(text)

        result = [
            label for label in self._labels if label.lower() in text.split()
        ]

        return result if result else [""]

    def _preprocess(
        self,
        text: str,
    ) -> str:
        """
        Preprocess output string before parsing result to solve common mistakes
        such as end-of-line presence and beginning and finishing with empty
        space.
        """

        return " ".join(
            text.lower()
            .replace("\n", " ")
            .strip()
            .split(self._labels_separator)
        )
