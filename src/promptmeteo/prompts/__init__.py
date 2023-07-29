from .base import BasePrompt
from .ner_prompt import NerPrompt
from .classification_prompt import ClassificationPrompt


NerPrompt.read_prompt_file(
    NerPrompt.PROMPT_EXAMPLE
    )

ClassificationPrompt.read_prompt_file(
    ClassificationPrompt.PROMPT_EXAMPLE
    )
