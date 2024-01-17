"""Custom selectors"""
from typing import Any, Dict, List, Optional, Type
import random
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings


def sorted_values(values: Dict[str, str]) -> List[Any]:
    """Return a list of values in dict sorted by key."""
    return [values[val] for val in sorted(values)]


class BalancedSemanticSamplesSelector(BaseExampleSelector, BaseModel):
    """Example selector that selects examples based on SemanticSimilarity in a balanced way."""

    vectorstore: VectorStore
    """A vectorstore per class is created"""
    selector_k_per_class: int = 2
    """Number of examples to select per class."""
    example_keys: Optional[List[str]] = None
    """Optional keys to filter examples to."""
    input_keys: Optional[List[str]] = None
    """Optional keys to filter input to. If provided, the search is based on
    the input variables instead of all variables."""
    vectorstore_kwargs: Optional[Dict[str, Any]] = None
    """Extra arguments passed to similarity_search function of the vectorstore."""
    class_list: List[str]
    """element of examples metadata which contains the class"""
    class_key: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> str:
        """Add new example to vectorstore."""
        if self.input_keys:
            string_example = " ".join(
                sorted_values({key: example[key] for key in self.input_keys})
            )
        else:
            string_example = " ".join(sorted_values(example))
        ids = self.vectorstore.add_texts([string_example], metadatas=[example])
        return ids[0]

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        class_list: List[str],
        class_key: str,
        embeddings: Embeddings,
        vectorstore_cls: Type[VectorStore],
        selector_k_per_class: int = 2,
        input_keys: Optional[List[str]] = None,
        **vectorstore_cls_kwargs: Any,
    ):
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            class_list: list of classes of the classification problem
            class_key: key which refers to category field in the example dictionary
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            selector_k_per_class: Number of examples to select per class
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        # for cl in class_list:
        # examples_subset = [eg for eg in examples if eg[class_key] == cl]
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]
        else:
            string_examples = [" ".join(sorted_values(eg)) for eg in examples]

        vectorstore = vectorstore_cls.from_texts(
            string_examples,
            embeddings,
            metadatas=examples,
            **vectorstore_cls_kwargs,
        )
        return cls(
            vectorstore=vectorstore,
            selector_k_per_class=selector_k_per_class,
            input_keys=input_keys,
            class_list=class_list,
            class_key=class_key,
        )

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        final_examples = []

        # Get the docs with the highest similarity.
        for cl in self.class_list:
            if self.input_keys:
                input_variables = {
                    key: input_variables[key] for key in self.input_keys
                }
            vectorstore_kwargs = self.vectorstore_kwargs or {}
            query = " ".join(sorted_values(input_variables))
            example_docs = self.vectorstore.similarity_search(
                query,
                k=self.selector_k_per_class,
                **vectorstore_kwargs,
                filter={self.class_key: cl},
            )
            # Get the examples from the metadata.
            # This assumes that examples are stored in metadata.
            examples = [dict(e.metadata) for e in example_docs]
            # If example keys are provided, filter examples to those keys.
            if self.example_keys:
                examples = [
                    {k: eg[k] for k in self.example_keys} for eg in examples
                ]
            final_examples = final_examples + examples

        random.shuffle(final_examples)
        return final_examples
