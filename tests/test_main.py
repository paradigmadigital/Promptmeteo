import os
import tempfile

import pytest

from promptmeteo import Promptmeteo


class TestPromptmeteo:


    def test_minimal_init(self):

        """
        Tests the exepected behaviour in a normal init.
        """

        model = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static')


    def test_minimal_init(self):

        """
        Tests the exepected behaviour in a normal init.
        """

        model = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static',
            model_provider_token='',
            prompt_labels=['positive','negative','neutral'],
            prompt_task_info="""
            Asume que estamos usando estas categorías con su significado
            subjetivo habitual: algo positivo puede describirse como bueno,
            de buena calidad, deseable, útil y satisfactorio; algo negativo
            puede describirse como malo, de mala calidad, indeseable, inútil
            o insatisfactorio; neutro es la clase que asignaremos a todo lo
            que no sea positivo o negativo.""",
            prompt_answer_format="""
            En tu respuesta incluye sólo el nombre de la clase, como una única
            palabra ({__LABELS__}), en minúscula, sin puntuación, y sin añadir
            ninguna otra afirmación o palabra.""",
            prompt_chain_of_thoughts="""
            Por favor argumenta tu respuesta paso a paso, explica por qué crees
            que está justificada tu elección final, y por favor asegúrate de
            que acabas tu explicación con el nombre de la clase que has elegido
            como la correcta, en minúscula y sin puntuación.""",
            selector_k=10,
            selector_algorithm='mmr',
            verbose=True)


    def test_wrong_init_parameters(self):

        """
        Tests load prompt text format
        """

        model = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static'
        ).read_prompt_file(
            '''
            TEMPLATE:
                "
                {__LABELS__}.

                {__TASK_INFO__}

                {__ANSWER_FORMAT__}

                {__CHAIN_OF_THOUGHTS__}
                "
            LABELS:
                ["positive", "negative"]

            TASK_INFO:
                "TEST_TASK_INFO"

            ANSWER_FORMAT:
                "TEST_ANSWER_FORMAT"

            CHAIN_OF_THOUGHTS:
                "TEST_CHAIN_OF_THOUGHTS"
            ''')

        prompt = model.task.prompt

        assert prompt.PROMPT_LABELS==["positive", "negative"]
        assert prompt.PROMPT_TASK_INFO=="TEST_TASK_INFO"
        assert prompt.PROMPT_ANSWER_FORMAT=="TEST_ANSWER_FORMAT"
        assert prompt.PROMPT_CHAIN_OF_THOUGHTS=="TEST_CHAIN_OF_THOUGHTS"


    def test_wrong_predict(self):

        """
        Test that the model perform predictions by calling the method
        `predict()`
        """

        with pytest.raises(Exception):
            pred = Promptmeteo(
                task_type='classification',
                model_provider_name='fake_llm',
                model_name='fake_static'
            ).predict("Wrong type this is expected to be a list")

        with pytest.raises(Exception):
            pred = Promptmeteo(
                task_type='classification',
                model_provider_name='fake_llm',
                model_name='fake_static'
            ).predict([1,2,3])


    def test_predict_without_train(self):

        """
        Test that the model perform predictions by calling the method
        `predict()`
        """

        pred = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static'
        ).predict(['positive'])

        assert pred==[['positive']]



    def test_wrong_train(self):

        """
        Test that the task property form Promptmeteo is trained after calling
        the method `fit()`
        """

        model = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static')

        with pytest.raises(Exception):

            model.train(
                examples = "Wrong type this is expected to be a list",
                annotations = "Wrong type this is expected to be a list"
            )

        with pytest.raises(Exception):

            model.train(
                examples = ["wrong element type: int", 1],
                annotations = ["wrong element type: boo", True]
            )

        with pytest.raises(Exception):

            model.train(
                examples = ["list", "with", "four", "items"],
                annotations = ["list", "with", "three"],
            )


    def test_correct_train(self):

        """
        Test that the task property form Promptmeteo is trained after calling
        the method `fit()`
        """

        model = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static')

        model = model.train(
            examples = ['estoy feliz', 'me da igual', 'no me gusta'],
            annotations = ['positive', 'neutral', 'negative']
        )

        assert model
        assert model.is_trained


    def test_predict_with_train(self):

        """
        Test that the model perform predictions by calling the method
        `predict()`
        """

        model = Promptmeteo(
            task_type='classification',
            model_provider_name='fake_llm',
            model_name='fake_static')

        model = model.train(
            examples = ['estoy feliz', 'me da igual', 'no me gusta'],
            annotations = ['positive', 'neutral', 'negative']
        )

        pred = model.predict(['positive'])

        assert pred==[['positive']]


    def test_save_model(self):

        model = Promptmeteo(
            task_type           = 'classification',
            model_provider_name = 'fake_llm',
            model_name          = 'fake_static',
            verbose             = True
        )

        # Try to save before training
        with pytest.raises(Exception) as error:
            model.save_model('model.meteo')
            assert error==(
                f'{model.__class__.__name__} error in `save_model()`. '
                f'You are trying to save a model that has non been trained. '
                f'Please, call `train()` function before'
            )

        model.train(
            examples = ['estoy feliz', 'me da igual', 'no me gusta'],
            annotations = ['positivo', 'neutral', 'negativo'],
        )

        # Use wrong file extension
        with pytest.raises(ValueError) as error:
            model_path = 'model.WRONG_EXTENSION'
            model.save_model(model_path)
            assert error==(
                f'{model.__class__.__name__} error in `save_model()`. '
                f'model_path="{model_path}" has a bad model name extension. '
                f'Model name must end with `.meteo` (i.e. `./model.meteo`)'
            )

        # Use non existing file path
        with pytest.raises(ValueError) as error:

            model_dir = 'WRONG_DIR_PATH'
            model.save_model(os.path.join(model_dir,'model.meteo'))
            assert error==(
                f'{model.__class__.__name__} error in `save_model()`. '
                f'directory {model_dir} does not exists.'
            )

        # Normal save
        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(os.path.join(tmp,'model.meteo'))
            assert os.path.exists(os.path.join(tmp,'model.meteo'))


    def test_load_model(self):

        model = Promptmeteo(
            task_type           = 'classification',
            model_provider_name = 'fake_llm',
            model_name          = 'fake_static',
            verbose             = True
        ).train(
            examples = ['estoy feliz', 'me da igual', 'no me gusta'],
            annotations = ['positivo', 'neutral', 'negativo'],
        )

        # Use wrong file extension
        with pytest.raises(ValueError) as error:
            model_path = 'model.WRONG_EXTENSION'
            model.load_model(model_path)
            assert error==(
                f'{model.__class__.__name__} error in `load_model()`. '
                f'model_path="{model_path}" has a bad model name extension. '
                f'Model name must end with `.meteo` (i.e. `./model.meteo`)'
            )

        # Use non existing file path
        with pytest.raises(ValueError) as error:

            model_dir = 'WRONG_DIR_PATH'
            model.load_model(os.path.join(model_dir,'model.meteo'))
            assert error==(
                f'{model.__class__.__name__} error in `load_model()`. '
                f'directory {model_dir} does not exists.'
            )

        # Normal load
        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(os.path.join(tmp,'model.meteo'))
            model.load_model(os.path.join(tmp,'model.meteo'))
