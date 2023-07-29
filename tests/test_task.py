import pytest

from promptmeteo.tasks import TaskTypes
from promptmeteo.models import BaseModel
from promptmeteo.tasks import TaskBuilderFactory


class TestTaskBuilder():


    def test_factory_method(self):

        for task_type in TaskTypes:
            TaskBuilderFactory.factory_method(
                task_type.value, verbose=False)

        for task_type in TaskTypes:
            TaskBuilderFactory.factory_method(
                task_type.value, verbose=True)

        with pytest.raises(Exception):
            TaskBuilderFactory.factory_method(
                'wrong_type')


    def test_build_model(self):

        for task_type in TaskTypes:
            task_builder = TaskBuilderFactory.factory_method(
                task_type.value
            ).build_model(
                model_name='fake_static',
                model_provider_name='fake_llm',
                model_provider_token=''
            )

            assert task_builder.task._model != None
            assert isinstance(task_builder.task._model, BaseModel)


    def test_build_prompt(self):

        for task_type in TaskTypes:
            task_builder = TaskBuilderFactory.factory_method(
                task_type.value
            ).build_prompt(
                prompt_template='{__TASK_INFO__}\n{__ANSWER_FORMAT__}',
                prompt_task_info="TASK_INFO",
                prompt_answer_format="ANSWER_FORMAT",
                prompt_chain_of_thoughts="CHAIN_OF_THOUGHTS"
            )

            assert task_builder.task._prompt != None


    def test_selector_prompt(self):

        for task_type in TaskTypes:
            with pytest.raises(Exception):
                task_builder = TaskBuilderFactory.factory_method(
                    task_type.value
                ).build_selector_by_train(
                    examples = ['estoy feliz', 'me da igual', 'no me gusta'],
                    annotations = ['positive', 'neutral', 'negative'],
                    selector_k=10,
                    selector_algorithm='mmr'
                )


        for task_type in TaskTypes:
            task_builder = TaskBuilderFactory.factory_method(
                task_type.value
            ).build_model(
                model_name='fake_static',
                model_provider_name='fake_llm',
                model_provider_token=''
            )

            task_builder.build_selector_by_train(
                examples = ['estoy feliz', 'me da igual', 'no me gusta'],
                annotations = ['positive', 'neutral', 'negative'],
                selector_k=10,
                selector_algorithm='mmr'
            )

            assert task_builder.task._selector != None
