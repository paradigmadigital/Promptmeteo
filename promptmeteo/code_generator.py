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

from .tasks import TaskTypes
from .tasks import TaskBuilder
from .base import BaseSupervised
from .tools import add_docstring_from


class CodeGenerator(BaseSupervised):

    """
    Code Generator Task.
    """

    @add_docstring_from(BaseSupervised.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Example
        -------

        >>> from promptmeteo import CodeGenerator

        >>> model = CodeGenerator(
        >>>     language='en',
        >>>     prompt_domain='python',
        >>>     model_provider_name='openai',
        >>>     model_name='text-davinci-003',
        >>>     model_provider_token=model_token,
        >>>     prompt_detail=[
        >>>         "add docstring in function definitions",
        >>>         "add argumment typing annotations"]
        >>>     )

        >>> pred=model.predict(['A function that receives the argument `foo`
        >>>     and prints it.'])

        def print_foo(foo: str):
            '''Prints the argument passed into the function'''
            print(foo)

        """

        super(CodeGenerator, self).__init__(**kwargs)

        #        if self.prompt_detail is None:
        #            raise ValueError(
        #                f"{self.__class__.__name__} error in initialization. "
        #                f"argument `prompt_detail` can not be None."
        #            )

        task_type = TaskTypes.CODE_GENERATION.value

        self._builder = TaskBuilder(
            language=self.language,
            task_type=task_type,
            verbose=self.verbose,
        )

        # Build model
        self._builder.build_model(
            model_name=self.model_name,
            model_provider_name=self.model_provider_name,
            model_provider_token=self.model_provider_token,
            model_params=self.model_params,
        )

        # Build prompt
        self._builder.build_prompt(
            model_name=self.model_name,
            prompt_domain=self.prompt_domain,
            prompt_labels=self.prompt_labels,
            prompt_detail=self.prompt_detail,
        )

        # Build parser
        self._builder.build_parser(
            prompt_labels=self.prompt_labels,
        )
