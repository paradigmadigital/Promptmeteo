import os
import tempfile

import pytest

from promptmeteo.tasks import Task
from promptmeteo.tasks import TaskBuilder
from promptmeteo.base import BaseUnsupervised


class TestBaseUnsupervised:
    def test_minimal_init(self):
        """
        Tests the exepected behaviour in a normal init.
        """

        model = BaseUnsupervised(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
        )

    def test_init_with_arguments(self):
        """
        Tests the exepected behaviour in a normal init.
        """

        model = BaseUnsupervised(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            model_provider_token="",
            prompt_domain="reseñas",
            prompt_detail="""
            Asume que estamos usando estas categorías con su significado
            subjetivo habitual: algo positivo puede describirse como bueno,
            de buena calidad, deseable, útil y satisfactorio; algo negativo
            puede describirse como malo, de mala calidad, indeseable, inútil
            o insatisfactorio; neutro es la clase que asignaremos a todo lo
            que no sea positivo o negativo.""",
            selector_k=10,
            selector_algorithm="mmr",
            verbose=True,
        )

        with pytest.raises(ValueError) as error:
            model = BaseUnsupervised(
                language="es",
                model_provider_name="fake-llm",
                model_name="fake-static",
                prompt_labels=["positive", "negative", "neutral"],
            )

            assert error.value.args[0] == (
                f"BaseUnsupervised can not be inicializated with "
                f"the argument `prompt_labels`."
            )

    def test_wrong_predict(self):
        """
        Test that the model perform predictions by calling the method
        `predict()`
        """

        with pytest.raises(ValueError) as error:
            pred = BaseUnsupervised(
                language="es",
                model_provider_name="fake-llm",
                model_name="fake-static",
            ).predict("Wrong type this is expected to be a list")

            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in function `predict()`. "
                f"Arguments `examples` and `annotations` are expected to be of "
                f"type `List[str]`. Instead they got: `{type(examples)}`"
            )

        with pytest.raises(ValueError) as error:
            pred = BaseUnsupervised(
                language="es",
                model_provider_name="fake-llm",
                model_name="fake-static",
            ).predict([1, 2, 3])

            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in function `predict()`. "
                f"Arguments `examples` are expected to be of type "
                f"`List[str]`. Some values seem no to be of type `str`."
            )

    def test_wrong_train(self):
        """
        Test that the task property form BaseUnsupervised is trained after
        calling the method `fit()`
        """

        model = BaseUnsupervised(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
        )

        with pytest.raises(ValueError) as error:
            model.train(
                examples="Wrong type this is expected to be a list",
            )

            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in function `train()`. "
                f"Arguments `examples` are expected to be of type `List[str]`."
                f"Instead they got: examples {type(examples)}"
            )

        with pytest.raises(ValueError) as error:
            model.train(
                examples=["wrong element type: int", 1],
            )

            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in function `train()`. "
                f"Arguments `examples` are expected to be of type `List[str]`."
                f"Some values seem no to be of type `str`."
            )

        with pytest.raises(TypeError) as error:
            model.train(
                examples=["1", "2", "3"],
                annotations=["one", "two", "three"],
            )

            assert error.value.args[0] == (
                f"TypeError: BaseUnsupervised.train() got an unexpected "
                f"keyword argument 'annotations'"
            )

    def test_load_model(self):
        model = BaseUnsupervised(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            verbose=True,
        )

        # Use wrong file extension
        with pytest.raises(ValueError) as error:
            model_path = "model.WRONG_EXTENSION"
            model.load_model(model_path)
            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in `load_model()`. "
                f'model_path="{model_path}" has a bad model name extension. '
                f"Model name must end with `.meteo` (i.e. `./model.meteo`)"
            )

        # Use non existing file path
        with pytest.raises(ValueError) as error:
            model_dir = "WRONG_DIR_PATH"
            model.load_model(os.path.join(model_dir, "model.meteo"))
            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in `load_model()`. "
                f"directory {model_dir} does not exists."
            )

    def test_save_model(self):
        model = BaseUnsupervised(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            verbose=True,
        )

        # Fake train
        model._is_trained = True

        # Try to save before training
        with pytest.raises(Exception) as error:
            model.save_model("model.meteo")
            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in `save_model()`. "
                f"You are trying to save a model that has non been trained. "
                f"Please, call `train()` function before"
            )

        # Use wrong file extension
        with pytest.raises(ValueError) as error:
            model_path = "model.WRONG_EXTENSION"
            model.save_model(model_path)
            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in `save_model()`. "
                f'model_path="{model_path}" has a bad model name extension. '
                f"Model name must end with `.meteo` (i.e. `./model.meteo`)"
            )

        # Use non existing file path
        with pytest.raises(ValueError) as error:
            model_dir = "WRONG_DIR_PATH"
            model.save_model(os.path.join(model_dir, "model.meteo"))
            assert error.value.args[0] == (
                f"{model.__class__.__name__} error in `save_model()`. "
                f"directory {model_dir} does not exists."
            )
