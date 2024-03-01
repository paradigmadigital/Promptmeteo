import pytest

from promptmeteo.parsers import ParserTypes
from promptmeteo.parsers import ParserFactory
from promptmeteo.parsers.dummy_parser import DummyParser
from promptmeteo.parsers.classification_parser import ClassificationParser
from promptmeteo.parsers.json_parser import JSONParser


class Testparsers:
    def test_parser_factory(self):
        for parser in ParserTypes:
            ParserFactory.factory_method(
                task_type=parser.value,
                prompt_labels=["true", "false"],
            )

        with pytest.raises(ValueError):
            ParserFactory.factory_method(
                task_type="WRONG PARSER TYPE",
                prompt_labels=["true", "false"],
            )

    def test_dummy_parser(self):
        parser = DummyParser(
            prompt_labels=["true", "false"],
        )

        assert ["blabla"] == parser.run("blabla")

    def test_classification_parser(self):
        parser = ClassificationParser(
            prompt_labels=["true", "false"],
        )

        assert [""] == parser.run("blabla")
        assert ["true"] == parser.run("true")
        assert ["true"] == parser.run("True")
        assert ["true"] == parser.run("blabla, true, blabla")
        assert ["true", "false"] == parser.run("true, false")
        
    def test_json_parser(self):
        parser = JSONParser(prompt_labels=["true","false"])
        
        wrong_json = """{
            "item1":[1,2,3,4],
            'item2':{
                "item2.1":"test item",
                'item2.2":["i211","i212",'i213']
            },
            "item3":"this is the third item"
        }"""
        
        correct_json = """{
            "item1":[1,2,3,4],
            "item2":{
                "item2.1":"test item",
                "item2.2":["i211","i212","i213"]
            },
            "item3":"this is the third item"
        }"""
        
        assert correct_json == parser.run(wrong_json)
        assert parser.run(correct_json) == parser.run(wrong_json)
        import json
        json.loads(parser.run(wrong_json))
        assert "" == parser.run("This is not a json format")
        assert "" == parser.run("{This is not a complete json")
