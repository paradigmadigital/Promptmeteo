import pytest

from promptmeteo.parsers import ParserTypes
from promptmeteo.parsers import ParserFactory
from promptmeteo.parsers.dummy_parser import DummyParser
from promptmeteo.parsers.classification_parser import ClassificationParser


class Testparsers():


    def test_parser_factory(self):

        for parser in ParserTypes:
            ParserFactory.factory_method(
                parser_type=parser.value,
                prompt_labels=['true','false'],
                prompt_labels_separator=',',
                prompt_chain_of_thoughts='TEST THOUGHTS'
            )

        with pytest.raises(Exception):
            ParserFactory.factory_method(
                parser_type='WRONG PARSER TYPE',
                prompt_labels=['true','false'],
                prompt_labels_separator=',',
                prompt_chain_of_thoughts='TEST THOUGHTS'
            )


    def test_dummy_parser(self):

        parser = DummyParser(
            prompt_labels=['true','false'],
            prompt_labels_separator=',',
            prompt_chain_of_thoughts='TEST THOUGHTS'
        )

        assert 'blabla' == parser.run('blabla')


    def test_classification_parser(self):

        parser = ClassificationParser(
            prompt_labels=['true','false'],
            prompt_labels_separator=',',
            prompt_chain_of_thoughts='TEST THOUGHTS'
        )

        assert [] == parser.run('blabla')
        assert ['true'] == parser.run('true')
        assert ['true'] == parser.run('True')
        assert ['true'] == parser.run('blabla, true, blabla')
        assert ['true', 'false'] == parser.run('true, false')
