import pytest

from langchain.embeddings import FakeEmbeddings

from promptmeteo.selector import SelectorFactory
from promptmeteo.selector import SelectorAlgorithms
from promptmeteo.selector.marginal_relevance_selector import MMRSelector
from promptmeteo.selector.semantic_similarity_selector import SimSelector


class TestSelectors():

    def test_selector_factory(self):

        for algorithm in SelectorAlgorithms:
            SelectorFactory.factory_method(
                selector_k = 1,
                selector_algorithm=algorithm.value,
                embeddings=FakeEmbeddings(size=64)
            )

        with pytest.raises(ValueError) as error:
            SelectorFactory.factory_method(
                selector_k = 1,
                selector_algorithm="WRONG_ALGORITHM_NAME",
                embeddings=FakeEmbeddings(size=64)
            )

            assert error==(
               f'`SelectorFactory` error in `factory_method()` . '
               f'{selector_algorithm} is not in the list of supported '
               f'providers: {[i.value for i in SelectorAlgorithms]}')


    def test_mmr_selector(self):

        selector = MMRSelector(
            embeddings=FakeEmbeddings(size=64),
            k=1
        )

        assert selector.SELECTOR is not None

        selector.train(
            examples = ['TEST_EXAMPLE'],
            annotations = ['TEST_ANNOTATION']
        )

        assert selector.template=="""
            Ejemplo: TEST_EXAMPLE
            Respuesta: TEST_ANNOTATION

            Ejemplo: {__INPUT__}
            Respuesta:""".replace(' '*4,'')[1:]


    def test_sim_selector(self):

        selector = SimSelector(
            embeddings=FakeEmbeddings(size=64),
            k=1
        )

        assert selector.SELECTOR is not None

        selector.train(
            examples = ['TEST_EXAMPLE'],
            annotations = ['TEST_ANNOTATION']
        )

        assert selector.template=="""
            Ejemplo: TEST_EXAMPLE
            Respuesta: TEST_ANNOTATION

            Ejemplo: {__INPUT__}
            Respuesta:""".replace(' '*4,'')[1:]
