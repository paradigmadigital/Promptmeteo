import os
import json
import gzip


module_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))


class DictionaryChecker:
    def __init__(self, language: str):
        try:
            file_name = f"{language}.json.gz"
            with gzip.open(os.path.join(module_dir, file_name)) as fin:
                self.lexicon = json.loads(fin.read())

        except FileNotFoundError:
            raise NotImplementedError(
                f"{self.__class__.__name__} error. Language `{language}`"
                f"is not implemented. Availables languages: sp, en"
            )

    def __call__(self, word: str):
        return word.lower() in self.lexicon or word.isnumeric()
