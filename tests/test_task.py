import pytest

from promptmeteo.models import BaseModel
from promptmeteo.tasks import TaskBuilder

task_types = ["classification"]


class TestTaskBuilder:
    def test_build_model(self):
        for task_type in task_types:
            task_builder = TaskBuilder(task_type).build_model(
                model_name="fake-static",
                model_provider_name="fake-llm",
                model_provider_token="",
            )

            assert task_builder.task._model is not None
            assert isinstance(task_builder.task._model, BaseModel)

    def test_build_prompt(self):
        for task_type in task_types:
            task_builder = TaskBuilder(task_type).build_prompt(
                language="es",
                task_type=task_type,
                model_name="fake-static",
                prompt_domain="TEST_DOMAIN",
                prompt_labels=["LABELS_1", "LABEL_2"],
                prompt_detail="TEST_DETAIL",
            )

            assert task_builder.task._prompt is not None

    def test_selector_prompt(self):
        for task_type in task_types:
            with pytest.raises(Exception):
                TaskBuilder(task_type).build_selector_by_train(
                    examples=["estoy feliz", "me da igual", "no me gusta"],
                    annotations=["positive", "neutral", "negative"],
                    selector_k=10,
                    selector_algorithm="mmr",
                )

        for task_type in task_types:
            task_builder = TaskBuilder(task_type).build_model(
                model_name="fake-static",
                model_provider_name="fake-llm",
                model_provider_token="",
            )

            task_builder.build_selector_by_train(
                examples=["estoy feliz", "me da igual", "no me gusta"],
                annotations=["positive", "neutral", "negative"],
                selector_k=10,
                selector_algorithm="mmr",
            )

            assert task_builder.task._selector is not None
