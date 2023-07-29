from enum import Enum
from typing import List
from typing import Optional

from .base import BaseTaskBuilder


class TaskTypes(Enum):

    TASK_1 = "classification"
    TASK_2 = "ner"


class TaskBuilderFactory():

    @staticmethod
    def factory_method(
        task_type   : str,
        task_labels : Optional[List[str]] = [''],
        verbose     : bool = False
    ) -> BaseTaskBuilder:

        if task_type == TaskTypes.TASK_1.value:
            from .classification_task import ClassificationTaskBuilder
            builder = ClassificationTaskBuilder

        elif task_type == TaskTypes.TASK_2.value:
            from .ner_task import NerTaskBuilder
            builder = NerTaskBuilder

        else:
             raise ValueError(
                f"{task_type} is not in the list of supported tasks: "
                f"{[i.value for i in TaskTypes]}"
                )

        return builder(
            task_labels=task_labels,
            verbose=verbose)
