import pytest

from promptmeteo.models import ModelFactory
from promptmeteo.models import ModelProvider


class TestModels():

    def test_model_factory(self):

        for model_provider_name in ModelProvider:
            with pytest.raises(ValueError) as error:

                ModelFactory.factory_method(
                    model_name           = 'TEST_NAME',
                    model_provider_name  = model_provider_name.value,
                    model_provider_token = 'TEST_TOKEN',
                    model_params         ={'TEST_PARAMS': True},
                )

                assert error!=(
                f'{model_provider_name} is not in the list of supported '
                f'providers: {[i.value for i in ModelProvider]}')


        for model_provider_name in ModelProvider:
            with pytest.raises(ValueError) as error:

                ModelFactory.factory_method(
                    model_name           = 'TEST_NAME',
                    model_provider_name  = 'WRONG_PROVIDER_NAME',
                    model_provider_token = 'TEST_TOKEN',
                    model_params         ={'TEST_PARAMS': True},
                )

                assert error==(
                f'{model_provider_name} is not in the list of supported '
                f'providers: {[i.value for i in ModelProvider]}')


    def test_model_openai(self):

        from promptmeteo.models.openai import ModelNames
        from promptmeteo.models.openai import OpenAILLM

        for model_name in ModelNames:
            OpenAILLM(
                model_name = model_name.value,
                model_params = {},
                model_provider_token = 'TEST_TOKEN')

        with pytest.raises(ValueError) as error:
            OpenAILLM(
                model_name = 'WRONG_NAME',
                model_params = {},
                model_provider_token = 'TEST_TOKEN')

    def test_model_fakellm(self):

        from promptmeteo.models.fake_llm import ModelNames
        from promptmeteo.models.fake_llm import FakeLLM

        for model_name in ModelNames:
            FakeLLM(
                model_name = model_name.value,
                model_params = {},
                model_provider_token = 'TEST_TOKEN')

        with pytest.raises(ValueError) as error:
            FakeLLM(
                model_name = 'WRONG_NAME',
                model_params = {},
                model_provider_token = 'TEST_TOKEN')


    def test_model_hf_hub_api(self):

        from promptmeteo.models.hf_hub_api import ModelNames
        from promptmeteo.models.hf_hub_api import HFHubApiLLM

        for model_name in ModelNames:
            HFHubApiLLM(
                model_name = model_name.value,
                model_params = {},
                model_provider_token = 'TEST_TOKEN')

        with pytest.raises(ValueError) as error:
            HFHubApiLLM(
                model_name = 'WRONG_NAME',
                model_params = {},
                model_provider_token = 'TEST_TOKEN')


    def test_model_hf_pipeline(self):

        from promptmeteo.models.hf_pipeline import ModelNames
        from promptmeteo.models.hf_pipeline import HFPipelineLLM

        with pytest.raises(ValueError) as error:
            HFPipelineLLM(
                model_name = 'WRONG_NAME',
                model_params = {},
                model_provider_token = 'TEST_TOKEN')
