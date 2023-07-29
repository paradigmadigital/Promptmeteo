import pytest
from langchain import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

from promptmeteo.prompts import BasePrompt


class TestPrompts:


    def test_minimal_init(self):

        """
        Tests the exepected behaviour in a normal init.
        """

        prompt = BasePrompt()


    def test_arguments_init(self):

        """
        Tests the exepected behaviour in a normal init.
        """

        prompt = BasePrompt(
            prompt_labels=['TEST_LABEL_1', 'TEST_LABEL_2'],
            prompt_task_info="TASK INFO",
            prompt_answer_format="RETURN LABELS: ({__LABELS__})",
            prompt_chain_of_thoughts="TASK EXPLAINED"
        )


        assert 'TEST_LABEL_1' in prompt.template
        assert 'TEST_LABEL_2' in prompt.template


    def test_update_prompt(self):

        """
        Tests load prompt text format
        """

        return_none = BasePrompt.read_prompt_file(
            '''
            TEMPLATE:
                "
                {__LABELS__}.

                {__TASK_INFO__}

                {__ANSWER_FORMAT__}

                {__CHAIN_OF_THOUGHTS__}
                "
            LABELS:
                ["true","false"]

            TASK_INFO:
                "TEST_TASK_INFO"

            ANSWER_FORMAT:
                "TEST_ANSWER_FORMAT"

            CHAIN_OF_THOUGHTS:
                "TEST_CHAIN_OF_THOUGHTS"
            ''')

        prompt = BasePrompt(
            prompt_labels=['TEST_LABEL_1', 'TEST_LABEL_2'],
            prompt_task_info="TASK INFO",
            prompt_answer_format="RETURN LABELS: ({__LABELS__})",
            prompt_chain_of_thoughts="TASK EXPLAINED"
        )

        assert return_none==None
        assert prompt.template == \
            " TEST_LABEL_1, TEST_LABEL_2.\n" + \
            "TASK INFO\nRETURN LABELS: (TEST_LABEL_1, TEST_LABEL_2)\n" + \
            "TASK EXPLAINED "
