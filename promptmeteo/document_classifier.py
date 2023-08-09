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
from typing import Dict
from typing import Optional

from .base import Base
from .tasks import TaskBuilder


class DocumentClassifier(Base):

    """
    DocumentClassifier Task
    """

    def __init__(
        self,
        model_name: str,
        model_provider_name: str,
        model_provider_token: Optional[str] = None,
        model_params: Optional[Dict] = {},
        language: str = "es",
        prompt_domain: Optional[str] = "",
        prompt_labels: List[str] = [],
        prompt_detail: Optional[str] = None,
        selector_k: int = 10,
        selector_algorithm: str = "mmr",
        verbose: bool = False,
    ) -> None:
        """
        Returns
        -------

        None

        Example
        -------

        >>> from promptmeteo import DocumentClassifier

        >>> clf = DocumentClassifier(
        >>>     task_type='classification',
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

        task_type = "classification"

        self._builder = TaskBuilder(task_type=task_type, verbose=verbose)

        # Build model
        self._builder.build_model(
            model_name=model_name,
            model_provider_name=model_provider_name,
            model_provider_token=model_provider_token,
            model_params=model_params,
        )

        # Build prompt
        self._builder.build_prompt(
            language=language,
            task_type=task_type,
            model_name=model_name,
            prompt_domain=prompt_domain,
            prompt_labels=prompt_labels,
            prompt_detail=prompt_detail,
        )

        # Build parser
        self._builder.build_parser(
            task_type=task_type,
            prompt_labels=prompt_labels,
        )

        # Selector config
        self._selector_k = selector_k
        self._selector_algorithm = selector_algorithm
