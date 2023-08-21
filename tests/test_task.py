import pytest

from promptmeteo.models import BaseModel
from promptmeteo.tasks import TaskBuilder
from promptmeteo.selector import SelectorTypes
from promptmeteo.selector.base import SelectorAlgorithms

task_types = ["classification", "qa"]


class TestTaskBuilder:
    def test_build_model(self):
        for task_type in task_types:
            task_builder = TaskBuilder(
                language="es",
                task_type=task_type,
            ).build_model(
                model_name="fake-static",
                model_provider_name="fake-llm",
                model_provider_token="",
            )

            assert task_builder.task.model is not None
            assert isinstance(task_builder.task.model, BaseModel)

    def test_build_prompt(self):
        for task_type in task_types:
            task_builder = TaskBuilder(
                language="es",
                task_type=task_type,
            ).build_prompt(
                model_name="fake-static",
                prompt_domain="TEST_DOMAIN",
                prompt_labels=["LABELS_1", "LABEL_2"],
                prompt_detail="TEST_DETAIL",
            )

            assert task_builder.task.prompt is not None

    def test_selector_prompt(self):
        for task_type in task_types:
            with pytest.raises(Exception):
                TaskBuilder(
                    language="es",
                    task_type=task_type,
                ).build_selector_by_train(
                    examples=["estoy feliz", "me da igual", "no me gusta"],
                    annotations=["positive", "neutral", "negative"],
                    selector_k=10,
                    selector_type="supervised",
                    selector_algorithm="mmr",
                )

        for task_type in task_types:
            for selector_algorithm in SelectorAlgorithms:
                task_builder = TaskBuilder(
                    language="es",
                    task_type=task_type,
                ).build_model(
                    model_name="fake-static",
                    model_provider_name="fake-llm",
                    model_provider_token="",
                )

                # SUPERVISED TASK
                selector_type = SelectorTypes.SUPERVISED.value
                task_builder.build_selector_by_train(
                    examples=["text1", "text2", "text3"],
                    annotations=["1", "0", "1"],
                    selector_k=10,
                    selector_type=selector_type,
                    selector_algorithm=selector_algorithm.value,
                )

                # UNSUPERVISED TASK
                selector_type = SelectorTypes.UNSUPERVISED.value
                task_builder.build_selector_by_train(
                    examples=["text1", "text2", "text3"],
                    annotations=None,
                    selector_k=10,
                    selector_type=selector_type,
                    selector_algorithm=selector_algorithm.value,
                )

            assert task_builder.task.selector is not None
