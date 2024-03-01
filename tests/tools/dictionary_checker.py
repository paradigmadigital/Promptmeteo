import os
import json
import gzip


module_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))


class DictionaryChecker:
    ADDED_WORDS = {
        "en": {"openapi": 1, "api": 1, "schema": 1, "schemas": 1},
        "es": {"openapi": 1, "api": 1, "sample":1, "examples":1, "generes":1, "human":1, "assistant":1},
    }

    def __init__(self, language: str):
        try:
            file_name = f"{language}.json.gz"
            with gzip.open(os.path.join(module_dir, file_name)) as fin:
                self.lexicon = json.loads(fin.read())
                self.lexicon.update(self.ADDED_WORDS[language])

        except FileNotFoundError:
            raise NotImplementedError(
                f"{self.__class__.__name__} error. Language `{language}`"
                f"is not implemented. Availables languages: sp, en"
            )

    def __call__(self, word: str):
        return word.lower() in self.lexicon or word.isnumeric()
