import os
import tempfile

import pytest

from promptmeteo import DocumentClassifier


class TestDocumentClassifier:
    def test_correct_train(self):
        """
        Test that the task property form DocumentClassifier is trained
        after calling the method `fit()`
        """

        model = DocumentClassifier(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
        )

        model = model.train(
            examples=["estoy feliz", "me da igual", "no me gusta"],
            annotations=["positive", "neutral", "negative"],
        )

        assert model
        assert model.is_trained
        assert model.prompt_labels == list(
            set(["positive", "neutral", "negative"])
        )

    def test_predict_with_train(self):
        """
        Test that the model perform predictions by calling the method
        `predict()`
        """

        model = DocumentClassifier(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
        )

        model = model.train(
            examples=["estoy feliz", "me da igual", "no me gusta"],
            annotations=["positive", "neutral", "negative"],
        )

        pred = model.predict(["positive"])

        assert pred == [["positive"]]

    def test_save_model(self):
        model = DocumentClassifier(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            verbose=True,
        )

        model.train(
            examples=["estoy feliz", "me da igual", "no me gusta"],
            annotations=["positivo", "neutral", "negativo"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(os.path.join(tmp, "model.meteo"))
            assert os.path.exists(os.path.join(tmp, "model.meteo"))

    def test_load_model(self):
        model = DocumentClassifier(
            language="es",
            model_provider_name="fake-llm",
            model_name="fake-static",
            verbose=True,
        ).train(
            examples=["estoy feliz", "me da igual", "no me gusta"],
            annotations=["positivo", "neutral", "negativo"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(os.path.join(tmp, "model.meteo"))
            model.load_model(os.path.join(tmp, "model.meteo"))
