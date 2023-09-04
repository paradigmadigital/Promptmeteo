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
from .tasks import TaskBuilder
from .base import BaseSupervised
from .tools import add_docstring_from


class DocumentClassifier(BaseSupervised):

    """
    DocumentClassifier Task
    """

    @add_docstring_from(BaseSupervised.__init__)
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Example
        -------

        >>> from promptmeteo import DocumentClassifier

        >>> clf = DocumentClassifier(
        >>>     model_provider_name='hf_pipeline',
        >>>     model_name='google/flan-t5-small',
        >>>     prompt_labels=['positive','negative','neutral'],
        >>> )

        >>> clf.train(
        >>>     examples = ['estoy feliz', 'me da igual', 'no me gusta'],
        >>>     annotations = ['positive', 'neutral', 'negative']
        >>> )

        >>> clf.predict(['que guay!!'])

        [['positive']]

        """

        super(DocumentClassifier, self).__init__(**kwargs)

        task_type = TaskTypes.CLASSIFICATION.value

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

    @add_docstring_from(BaseSupervised.train)
    def train(
        self,
        examples: List[str],
        annotations: List[str],
    ) -> Self:
        """ """

        super(DocumentClassifier, self).train(
            examples=examples, annotations=annotations
        )

        if not self.prompt_labels:
            self.prompt_labels = list(set(annotations))
            self._builder.build_parser(
                prompt_labels=self.prompt_labels,
            )

        return self
