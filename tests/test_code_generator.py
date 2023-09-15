import os
import tempfile

import pytest

from promptmeteo import CodeGenerator


class TestCodeGenerator:
    def test_correct_train(self):
        """
        Test that the task property form CodeGenerator is trained
        after calling the method `fit()`
        """

        model = CodeGenerator(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            prompt_detail=["add docstring in function definitions"],
        )

        model = model.train(
            examples=["Function with the argument `foo` that prints it"],
            annotations=[
                '''
                def print_foo(foo: str):
                """This function prints the value passed in as an argument.

                Args:        foo (str): The value to be printed.
                """
                print(foo)'''.replace(
                    " " * 16, ""
                )
            ],
        )

        assert model
        assert model.is_trained

    def test_predict_with_train(self):
        """
        Test that the model perform predictions by calling the method
        `predict()`
        """

        model = CodeGenerator(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            prompt_detail=["add docstring in function definitions"],
        )

        model = model.train(
            examples=["Function with the argument `foo` that prints it"],
            annotations=[
                '''
                def print_foo(foo: str):
                """This function prints the value passed in as an argument.

                Args:        foo (str): The value to be printed.
                """
                print(foo)'''.replace(
                    " " * 16, ""
                )
            ],
        )

        pred = model.predict(["positive"])

        assert pred == [["positive"]]

    def test_save_model(self):
        model = CodeGenerator(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            verbose=True,
        )

        model.train(
            examples=["Function with the argument `foo` that prints it"],
            annotations=[
                '''
                def print_foo(foo: str):
                """This function prints the value passed in as an argument.

                Args:        foo (str): The value to be printed.
                """
                print(foo)'''.replace(
                    " " * 16, ""
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(os.path.join(tmp, "model.meteo"))
            assert os.path.exists(os.path.join(tmp, "model.meteo"))

    def test_load_model(self):
        model = CodeGenerator(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            verbose=True,
        ).train(
            examples=["Function with the argument `foo` that prints it"],
            annotations=[
                '''
                def print_foo(foo: str):
                """This function prints the value passed in as an argument.

                Args:        foo (str): The value to be printed.
                """
                print(foo)'''.replace(
                    " " * 16, ""
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(os.path.join(tmp, "model.meteo"))
            model.load_model(os.path.join(tmp, "model.meteo"))
