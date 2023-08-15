from unittest.mock import MagicMock

import pytest

from promptmeteo.models import ModelFactory
from promptmeteo.models import ModelProvider


class TestModels:
    def test_model_factory(self):
        invalid_provider = (
            "ModelFactory error in `factory_method()`. "
            "WRONG_PROVIDER_NAME is not in the list of supported "
            f"providers: {[i.value for i in ModelProvider]}"
        )
        for model_provider_name in ModelProvider:
            with pytest.raises(ValueError) as error:
                ModelFactory.factory_method(
                    model_name="TEST_NAME",
                    model_provider_name=model_provider_name.value,
                    model_provider_token="TEST_TOKEN",
                    model_params={"TEST_PARAMS": True},
                )

            assert error.value.args[0] != invalid_provider

        with pytest.raises(ValueError) as error:
            ModelFactory.factory_method(
                model_name="TEST_NAME",
                model_provider_name="WRONG_PROVIDER_NAME",
                model_provider_token="TEST_TOKEN",
                model_params={"TEST_PARAMS": True},
            )

        assert error.value.args[0] == invalid_provider

    def test_model_openai(self):
        from promptmeteo.models.openai import ModelTypes
        from promptmeteo.models.openai import OpenAILLM

        for model_name in ModelTypes:
            OpenAILLM(
                model_name=model_name.value,
                model_params={},
                model_provider_token="TEST_TOKEN",
            )

        with pytest.raises(ValueError) as error:
            OpenAILLM(
                model_name="WRONG_NAME",
                model_params={},
                model_provider_token="TEST_TOKEN",
            )
        invalid_provider = (
            "`model_name`=WRONG_NAME not in supported model names: "
            f"{[i.name for i in ModelTypes]}"
        )
        assert error.value.args[0] == invalid_provider

    def test_model_fakellm(self):
        from promptmeteo.models.fake_llm import ModelTypes
        from promptmeteo.models.fake_llm import FakeLLM

        for model_name in ModelTypes:
            FakeLLM(
                model_name=model_name.value,
                model_params={},
                model_provider_token="TEST_TOKEN",
            )

        with pytest.raises(ValueError) as error:
            FakeLLM(
                model_name="WRONG_NAME",
                model_params={},
                model_provider_token="TEST_TOKEN",
            )
        invalid_provider = (
            "FakeLLM error creating object. "
            "WRONG_NAME is not in the list of supported FakeLLMS: "
            f"{[i.value for i in ModelTypes]}"
        )
        assert error.value.args[0] == invalid_provider

    def test_model_hf_hub_api(self, mocker):
        from promptmeteo.models.hf_hub_api import ModelTypes
        from promptmeteo.models.hf_hub_api import HFHubApiLLM

        mock = MagicMock()
        mocker.patch(
            "huggingface_hub.inference_api.InferenceApi.__call__", mock
        )

        for model_name in ModelTypes:
            HFHubApiLLM(
                model_name=model_name.value,
                model_params={},
                model_provider_token="TEST_TOKEN",
            )

        with pytest.raises(ValueError) as error:
            HFHubApiLLM(
                model_name="WRONG_NAME",
                model_params={},
                model_provider_token="TEST_TOKEN",
            )

        invalid_provider = (
            "`model_name`=WRONG_NAME not in supported model names: "
            f"{[i.value for i in ModelTypes]}"
        )
        assert error.value.args[0] == invalid_provider

    def test_model_hf_pipeline(self):
        from promptmeteo.models.hf_pipeline import ModelTypes
        from promptmeteo.models.hf_pipeline import HFPipelineLLM

        with pytest.raises(ValueError) as error:
            HFPipelineLLM(
                model_name="WRONG_NAME",
                model_params={},
                model_provider_token="TEST_TOKEN",
            )

        invalid_provider = (
            "`model_name`=WRONG_NAME not in supported model names: "
            f"{[i.value for i in ModelTypes]}"
        )
        assert error.value.args[0] == invalid_provider
