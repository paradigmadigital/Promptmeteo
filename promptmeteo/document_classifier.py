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

from .tasks import TaskTypes
from .base import BaseSupervised
from .tools import add_docstring_from


class DocumentClassifier(BaseSupervised):
    """
    DocumentClassifier Task

    This class represents a model for classifying documents into predefined categories.

    Example
    -------
    >>> from promptmeteo import DocumentClassifier
    >>> clf = DocumentClassifier(
    ...     model_provider_name='hf_pipeline',
    ...     model_name='google/flan-t5-small',
    ...     prompt_labels=['positive','negative','neutral'],
    ... )

    >>> clf.train(
    ...     examples = ['estoy feliz', 'me da igual', 'no me gusta'],
    ...     annotations = ['positive', 'neutral', 'negative']
    ... )

    >>> clf.predict(['que guay!!'])

    >>> [['positive']]

    """

    TASK_TYPE = TaskTypes.CLASSIFICATION.value

    @add_docstring_from(BaseSupervised.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize the DocumentClassifier model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.
        """

        super(DocumentClassifier, self).__init__(**kwargs)

    @add_docstring_from(BaseSupervised.train)
    def train(
        self,
        examples: List[str],
        annotations: List[str],
    ) -> Self:
        """
        Trains the DocumentClassifier model.

        Parameters
        ----------
        examples : List[str]
            List of document examples.
        annotations : List[str]
            List of corresponding annotations.

        Returns
        -------
        Self
        """

        super(DocumentClassifier, self).train(
            examples=examples, annotations=annotations
        )

        if not self.prompt_labels:
            self.prompt_labels = list(set(annotations))
            self.builder.build_parser(
                prompt_labels=self.prompt_labels,
            )

        return self
