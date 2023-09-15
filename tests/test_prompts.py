import os
import yaml
import pytest
from tests.tools import DictionaryChecker

from promptmeteo.prompts import BasePrompt
from promptmeteo.prompts import PromptFactory
from promptmeteo.prompts import module_dir


class TestPrompts:
    def test_prompt_factory(self):
        for model_name, language, task_type in [
            ["fake-static", "es", "ner"],
            ["fake-static", "es", "classification"],
        ]:
            prompt = PromptFactory.factory_method(
                language=language,
                task_type=task_type,
                model_name=model_name,
                prompt_domain="TEST DOMAIN",
                prompt_labels=["true", "false"],
                prompt_detail="TEST PROMPT DETAIL",
            )

    def test_prompt_factory_exception(self):
        with pytest.raises(ValueError):
            PromptFactory.factory_method(
                language="WRONG LANGUAGE",  # WRONG
                task_type="ner",
                model_name="fake-static",
                prompt_domain="TEST DOMAIN",
                prompt_labels=["true", "false"],
                prompt_detail="TEST PROMPT DETAIL",
            )

        with pytest.raises(ValueError):
            PromptFactory.factory_method(
                language="es",
                task_type="WRONG TASK TYPE",  # WRONG
                model_name="fake-static",
                prompt_domain="TEST DOMAIN",
                prompt_labels=["true", "false"],
                prompt_detail="TEST PROMPT DETAIL",
            )

        with pytest.raises(ValueError):
            PromptFactory.factory_method(
                language="es",
                task_type="ner",
                model_name="WRONG MODEL NAME",  # WRONG
                prompt_domain="TEST DOMAIN",
                prompt_labels=["true", "false"],
                prompt_detail="TEST PROMPT DETAIL",
            )

    def test_base_prompt(self):
        """
        Test BasePrompt behavior.
        """

        BasePrompt.read_prompt(BasePrompt.PROMPT_EXAMPLE)

        BasePrompt(
            prompt_domain="TEST DOMAIN",
            prompt_labels=["true", "false"],
            prompt_detail="TEST PROMPT DETAIL",
        )

    def test_base_prompt_exception(self):
        """
        Tests load prompt text format.
        """

        with pytest.raises(ValueError) as error:
            BasePrompt.read_prompt(
                """
                WRONG_TEMPLATE_KEY:
                    wrong template key

                ANOTHER_WRONG_TEMPLATE_KEY:
                    wrong template key again
                """
            )

        assert error.value.args[0] == (
            f"`BasePrompt` error `read_prompt()`. "
            f"The expected keys are {yaml.load(BasePrompt.PROMPT_EXAMPLE, Loader=yaml.FullLoader)}"
        )

    def test_prompts_naming(self):
        """
        Test that the hierrachycal structure of the prompts is correct.
        """

        prompt_files = [
            file_name
            for file_name in os.listdir(module_dir)
            if file_name.endswith(".prompt")
        ]

        assert all([len(file.split("_")) == 3 for file in prompt_files])

    def test_prompts_format(self):
        """
        Test that all the prompts from the project follow the expected
        structure.
        """

        params = [
            file_name.replace(".prompt", "").split("_")
            for file_name in os.listdir(module_dir)
            if file_name.endswith(".prompt")
        ]

        for model_name, language, task_type in params:
            prompt = PromptFactory.factory_method(
                language=language,
                task_type=task_type,
                model_name=model_name,
                prompt_domain="",
                prompt_labels=["0", "1"],
                prompt_detail="",
            )

    def test_prompts_spelling(self):
        """
        Test that all the prompts are written correctly.
        """

        params = [
            file_name.replace(".prompt", "").split("_")
            for file_name in os.listdir(module_dir)
            if file_name.endswith(".prompt")
        ]

        for model_name, language, task_type in params:
            text = PromptFactory.factory_method(
                language=language,
                task_type=task_type,
                model_name=model_name,
                prompt_domain="",
                prompt_labels=["0", "1"],
                prompt_detail="",
            ).template

            check = DictionaryChecker(language)

            for symbol in """!()-[]{};:'"\\,<>./?@#$%^&*_~""":
                text = text.replace(symbol, " ")

            for word in text.split():
                if not check(word):
                    raise Exception(
                        f"Error in {model_name}_{language}_{task_type}.prompt "
                        f"the word `{word}` is not in the dictionary"
                    )
