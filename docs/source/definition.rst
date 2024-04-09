🤔 What is this library for?
==============================

**TL;DR: Industrialize projects powered by LLMs easily.**

LLMs have the capability to address various tasks when provided with specific instructions in the form of input prompts. They can function as a "reasoning engine" for constructing applications. However, deploying and industrializing these applications poses significant challenges for two main reasons.

Firstly, prompts typically encapsulate application logic in their definition, yet they are treated merely as input arguments. This means that a poorly formulated prompt input has the potential to disrupt the application.

Secondly, crafting specific prompts for each task is not only a laborious task but also a complex one. Minor alterations in the input prompt can result in different outcomes, rendering them highly error-prone. Additionally, when composing prompts, considerations extend beyond the task itself to include factors such as the specific LLM being used, the model's capacity, and other relevant aspects.

🚀 How do we do it?
----------------------

**TL;DR: Treating prompts and code equally!!!**

Promptmeteo aims to address the aforementioned issues by dividing the prompt definition into two distinct parts: the task logic, coded within prompt templates, and the concrete problem, included as argument variables. Promptmeteo incorporates high-level objects for various tasks, implemented through `.py` and `.prompt` files.

🏠 Prebuilt tasks
^^^^^^^^^^^^^^^^^^

The project incorporates high-level objects designed to address various NLP tasks, including text classification, named entity recognition, and code generation. These objects only require configuration parameters for execution, eliminating the need to parse the output from the LLMs.

⚙️ Ease of Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Promptmeteo modules adhere to a model interface similar to Scikit-Learn. By defining an interface with independent methods for training, predicting, saving, and loading the model, Promptmeteo enables training in a separate pipeline from prediction. This facilitates the reuse of conventional ML pipelines for LLM projects.

📦 Model Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^

To enhance results, LLMs can incorporate examples in their prompts. Promptmeteo supports training with examples, and for reproducibility, the training process can be stored as a binary model artifact. This allows storing and reusing training results multiple times with new data. The training process stores embeddings from the input text in a vector database, such as FAISS.

⚙️ LLMs Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Promptmeteo integrates different LLMs through LangChain. This includes models that can be executed locally and remote API calls from providers like OpenAI and HuggingFace.

📄 Prompt Templating
^^^^^^^^^^^^^^^^^^^^^^^^^^

Establishing a concrete format for creating prompts in Promptmeteo (`.prompt`) not only facilitates programmatic use but also enables versioning of prompts. This approach aids in understanding changes when they occur and allows for the definition of code tests oriented toward prompt testing. This testing encompasses aspects such as validating language use and ensuring the prompt size is appropriate for the model.

.. code-block:: yaml

    TEMPLATE:
        "I need you to help me with a text classification task.
        {__PROMPT_DOMAIN__}
        {__PROMPT_LABELS__}

        {__CHAIN_THOUGHT__}
        {__ANSWER_FORMAT__}"

    PROMPT_DOMAIN:
        "The texts you will be processing are from the {__DOMAIN__} domain."

    PROMPT_LABELS:
        "I want you to classify the texts into one of the following categories:
        {__LABELS__}."

    PROMPT_DETAIL:
        ""

    CHAIN_THOUGHT:
        "Please provide a step-by-step argument for your answer, explain why you
        believe your final choice is justified."

    ANSWER_FORMAT:
        "In your response, include only the name of the class as a single word, in
        lowercase, without punctuation, and without adding any other statements or
        words."


📋 Current capacilities
----------------------------

✅ Available tasks
^^^^^^^^^^^^^^^^^^

The current available tasks in Promptmeteo are:

.. list-table:: Tasks
    :header-rows: 1

    * - task type
      - description
    * - `DocumentQA`
      - Document-level question answering
    * - `DocumentClassifier`
      - Document-level classification
    * - `CodeGenerator`
      - Code generation
    * - `ApiGenerator`
      - API REST generation
    * - `ApiFormatter`
      - API REST correction
    * - `Summarizer`
      - Text summarization

✅ Available Models
^^^^^^^^^^^^^^^^^^^^^^^^

The current available `model_name` and `language` values are:

.. list-table:: Models
    :header-rows: 1

    * - provider
      - name
      - languages
    * - openai
      - gpt-3.5-turbo-16k
      - es, en
    * - azure
      - gpt-3.5-turbo-16k
      - es, en
    * - hf_hub_api
      - google/flan-t5-xxl
      - es, en
    * - hf_pipeline
      - google/flan-t5-small
      - es, en
    * - google
      - text-bison
      - es, en
    * - google
      - text-bison@001
      - es, en
    * - google
      - text-bison-32k
      - es, en
    * - bedrock
      - anthropic.claude-v2
      - es, en
