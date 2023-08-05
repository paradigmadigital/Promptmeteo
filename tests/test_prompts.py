import pytest

from promptmeteo.prompts import PromptTypes
from promptmeteo.prompts import PromptFactory

from promptmeteo.prompts import NerPrompt
from promptmeteo.prompts import BasePrompt
from promptmeteo.prompts import ClassificationPrompt


class TestPrompts:

    def test_prompt_factory(self):

        for prompt in PromptTypes:
            PromptFactory.factory_method(
                prompt_type=prompt.value,
                prompt_domain='TEST DOMAIN',
                prompt_labels=['true','false'],
                prompt_detail='TEST PROMPT DETAIL'
            )

        with pytest.raises(ValueError):
            PromptFactory.factory_method(
                prompt_type='WRONG PARSER TYPE',
                prompt_domain='TEST DOMAIN',
                prompt_labels=['true','false'],
                prompt_detail='TEST PROMPT DETAIL'
            )

    def test_minimal_init(self):

        """
        Tests the exepected behaviour in a normal init.
        """

        prompt = ClassificationPrompt()


    def test_arguments_init(self):

        """
        Tests the exepected behaviour in a normal init.
        """

        prompt = ClassificationPrompt(
            prompt_domain= 'TEST_DOMAIN',
            prompt_labels=['TEST_LABEL_1', 'TEST_LABEL_2'],
            prompt_detail= 'TEST_DETAIL',
        )


        assert 'TEST_DOMAIN' in prompt.template
        assert 'TEST_LABEL_1' in prompt.template
        assert 'TEST_LABEL_2' in prompt.template


    def test_update_prompt(self):

        """
        Tests load prompt text format
        """

        with pytest.raises(ValueError):
            ClassificationPrompt.read_prompt(
                '''
                WRONG_TEMPLATE_KEY:
                    wrong template key

                ANOTHER_WRONG_TEMPLATE_KEY:
                    wrong template key again
                ''')
