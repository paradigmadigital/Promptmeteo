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
from .base import BaseUnsupervised
from .tools import add_docstring_from


class DocumentQA(BaseUnsupervised):
    """
    Question Answering over Documents Task

    This class represents a model for answering questions based on documents.

    Example
    -------
    >>> from promptmeteo import DocumentQA
    >>> clf = DocumentQA(
    ...     language='en',
    ...     model_provider_name='hf_pipeline',
    ...     model_name='google/flan-t5-small',
    ... )

    >>> clf.train(
    ...     examples = [
    ...     "The rain in spain is always in plain",
    ...     "The logarithm's limit is the limit's logarithm",
    ...     "To punish oppresors is clementy. To forgive them is cruelty"],
    ... )

    >>> clf.predict(['How is the rain in spain?'])

    >>> [['in plain']]
    """

    TASK_TYPE = TaskTypes.QA.value

    @add_docstring_from(BaseUnsupervised.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize the DocumentQA model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.
        """

        super(DocumentQA, self).__init__(**kwargs)
