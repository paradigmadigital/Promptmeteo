import pytest

from langchain.embeddings import FakeEmbeddings

from promptmeteo.selector import SelectorFactory
from promptmeteo.selector import SelectorTypes
from promptmeteo.selector.marginal_relevance_selector import MMRSelector
from promptmeteo.selector.semantic_similarity_selector import SimSelector


class TestSelectors:
    def test_selector_factory(self):
        for algorithm in SelectorTypes:
            SelectorFactory.factory_method(
                language="es",
                selector_k=1,
                selector_algorithm=algorithm.value,
                embeddings=FakeEmbeddings(size=64),
            )

        with pytest.raises(ValueError) as error:
            SelectorFactory.factory_method(
                language="es",
                selector_k=1,
                selector_algorithm="WRONG_ALGORITHM_NAME",
                embeddings=FakeEmbeddings(size=64),
            )

        assert error.value.args[0] == (
            f"`SelectorFactory` error in `factory_method()` . "
            f"WRONG_ALGORITHM_NAME is not in the list of supported "
            f"providers: {[i.value for i in SelectorTypes]}"
        )

    def test_mmr_selector(self):
        selector = MMRSelector(
            language="es",
            embeddings=FakeEmbeddings(size=64),
            k=1,
        )

        assert selector.SELECTOR is not None

        selector.train(
            examples=["TEST_EXAMPLE"], annotations=["TEST_ANNOTATION"]
        )

        assert (
            selector.template
            == """
            EJEMPLO: TEST_EXAMPLE
            RESPUESTA: TEST_ANNOTATION

            EJEMPLO: {__INPUT__}
            RESPUESTA: """.replace(
                " " * 4, ""
            )[
                1:
            ]
        )

    def test_sim_selector(self):
        selector = SimSelector(
            language="es",
            embeddings=FakeEmbeddings(size=64),
            k=1,
        )

        assert selector.SELECTOR is not None

        selector.train(
            examples=["TEST_EXAMPLE"], annotations=["TEST_ANNOTATION"]
        )
        expected_template = """
        EJEMPLO: TEST_EXAMPLE
        RESPUESTA: TEST_ANNOTATION

        EJEMPLO: {__INPUT__}
        RESPUESTA: """.replace(
            " " * 4, ""
        )[
            1:
        ]

        assert selector.template == expected_template
