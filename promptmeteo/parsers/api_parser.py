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

from .base import BaseParser, ParserException


class ApiParser(BaseParser):
    """
    Dummy parser, returns what it receives.
    """

    def run(
        self,
        text: str,
    ) -> str:
        """
        Given a response string from an LLM, returns the response expected for
        the task.
        """

        text = text.replace("{{", "{").replace("}}", "}")

        if "as an AI language model" in text:
            raise ParserException("There's no API definition in this response")

        pattern = re.findall(r"```[a-zA-Z]+?\n(.*?)```", text, re.DOTALL)
        text = pattern[0] if pattern else text

        pattern = re.findall(r"```(.*?)```", text, re.DOTALL)
        text = pattern[0] if pattern else text

        text = re.sub(r"([\s\S]*?)openapi:", "openapi:", text)

        return text.replace("```", "")
