![python-versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![code-lint](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/code_lint.yml/badge.svg)
![code-testing](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/code_testing.yml/badge.svg)
![publish-docker](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/publish_docker.yml/badge.svg)
![publish-pypi](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/publish_package.yml/badge.svg)
[![codecov](https://codecov.io/gh/paradigmadigital/Promptmeteo/branch/main/graph/badge.svg?token=KFJS6BGFH8)](https://codecov.io/gh/DelgadoPanadero/PromptMeteo)

# PromptMeteo 🔥🧔

**Promptmeteo** is a Python library for prompt engineering built over LangChain. It simplifies the utilization of large language models (LLMs) for various tasks through a low-code interface. To achieve this, Promptmeteo can employ different LLM models and dynamically generate prompts for specific tasks based on just a few configuration parameters.




&nbsp;

# 🤔 What is this library for?
**TL;DR: Industrialize projects powered by LLMs easily.**

LLMs have the capability to address various tasks when provided with specific instructions in the form of input prompts. They can function as a "reasoning engine" for constructing applications. However, deploying and industrializing these applications poses significant challenges for two main reasons.

Firstly, prompts typically encapsulate application logic in their definition, yet they are treated merely as input arguments. This means that a poorly formulated prompt input has the potential to disrupt the application.

Secondly, crafting concrete prompts for each task is not only a laborious task but also a complex one. Minor alterations in the input prompt can result in different outcomes, rendering them highly error-prone. Additionally, when composing prompts, considerations extend beyond the task itself to include factors such as the specific LLM being used, the model's capacity, and other relevant aspects.




&nbsp;
# 🚀 How do we solve it?

**TL;DR: Treating prompts and code equally!!!**

Promptmeteo aims to address the aforementioned issues by dividing the prompt definition into two distinct parts: the task logic, coded within prompt templates, and the concrete problem, included as argument variables. Promptmeteo incorporates high-level objects for various tasks, implemented through `.py` and `.prompt` files.

#### 🏠 Prebuilt tasks

The project incorporates high-level objects designed to address various NLP tasks, including text classification, named entity recognition, and code generation. These objects only require configuration parameters for execution, eliminating the need to parse the output from the LLMs.

#### ⚙️ Ease of Deployment

Promptmeteo modules adhere to a model interface similar to Scikit-Learn. By defining an interface with independent methods for training, predicting, saving, and loading the model, Promptmeteo enables training in a separate pipeline from prediction. This facilitates the reuse of conventional ML pipelines for LLM projects.

#### 📦 Model Artifacts

To enhance results, LLMs can incorporate examples in their prompts. Promptmeteo supports training with examples, and for reproducibility, the training process can be stored as a binary model artifact. This allows storing and reusing training results multiple times with new data. The training process stores embeddings from the input text in a vector database, such as FAISS.

#### ⚙️ LLMs Integration 

Promptmeteo integrates different LLMs through LangChain. This includes models that can be executed locally and remote API calls from providers like OpenAI and HuggingFace.

#### 📄 Prompt Templating

Establishing a concrete format for creating prompts in Promptmeteo (`.prompt`) not only facilitates programmatic use but also enables versioning of prompts. This approach aids in understanding changes when they occur and allows for the definition of code tests oriented toward prompt testing. This testing encompasses aspects such as validating language use and ensuring the prompt size is appropriate for the model.

```yaml
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
```


&nbsp;
# ⚡ Quick start

First of all, do not forget to configure providers credentials. Refer to the [configure credentials section](#⚙️-configure-credentials).

### ✨ Create the task
You can make a prediction directly indicating the model and calling the method `predict()`.

```python
from promptmeteo import DocumentClassifier

clf = DocumentClassifier(
        language            = 'en',
        model_provider_name = 'hf_pipeline',
        model_name          = 'google/flan-t5-small',
        prompt_labels       = ['positive', 'neutral', 'negative']
    )

clf.predict(['so cool!!'])
```

```shell
[['positive']]
```

### ✨ Train the task
But you can also include examples to improve the results by calling the method `train()`.


```python
clf = clf.train(
    examples    = ['i am happy', 'doesnt matter', 'I hate it'],
    annotations = ['positive', 'neutral', 'negative'],
)

clf.predict(['so cool!!'])
```

```shell
[['positive']]
```

### ✨ Save a trained task

Once the model is trained it can be saved locally...

```python
clf.save_model("hello_world.meteo")
```


### ✨ Load a trained task

... and loaded again to make new predictions.

```python
from promptmeteo import DocumentClassifier

clf = DocumentClassifier(
        language            = 'en',
        model_provider_name = 'hf_pipeline',
        model_name          = 'google/flan-t5-small',
    ).load_model("hello_world.meteo")

clf.predict(['so cool!!'])
```

```shell
[['positive']]
```

Models can also be loaded without instantiating the class by using `load_model()` as a function instead of a method:

```python
from promptmeteo import DocumentClassifier

clf = DocumentClassifier.load_model("hello_world.meteo")
```

###  ✨ Learn more

More examples can be seen in the directory [examples](./examples).

&nbsp;


## 🚗 Usage

### ⚙️ Configure credentials
Create a ```.env``` with the following variables depending on the LLM provider.

#### Google Cloud
First you should create a [Service Account](https://cloud.google.com/vertex-ai/docs/general/custom-service-account#configure) with the role: ``Vertex AI User.``

Once created, generate a key, store it locally and reference the path in the .env file: 

```shell
GOOGLE_CLOUD_PROJECT_ID="MY_GOOGLE_LLM_PROJECT_ID"
GOOGLE_APPLICATION_CREDENTIALS="PATH_TO_SERVICE_ACCOUNT_KEY_FILE.json"
```

#### OpenAI
Create your Secret API key in your User settings [page](https://platform.openai.com/account/api-keys).

Indicate the value of the key in your .env file:

```shell
OPENAI_API_KEY="MY_OPENAI_API_KEY"
```

You can also pass `openai_api_key` as a named parameter.

#### Hugging Face
Create Access Token in your User settings [page](https://huggingface.co/settings/tokens).

```shell
HUGGINGFACEHUB_API_TOKEN="MY_HF_API_KEY"
```

You can also pass `huggingfacehub_api_token` as a named parameter.

#### AWS Bedrock
Create your access keys in security credentials of your user in AWS.

Then write in the files ```~/.aws/config``` and ````~/.aws/credentials```` for Linux and MacOS or ````%USERPROFILE%\.aws\config```` and ````%USERPROFILE%\.aws\credentials```` for Windows:

In credentials:
```shell
[default]
aws_access_key_id = <YOUR_CREATED_AWS_KEY>
aws_secret_access_key = <YOUR_CREATED_AWS_SECRET_KEY>
```

In config:
```shell
[default]
region = <AWS_REGION>
```

### ⚙️ Install locally

```shell
make setup
```

### ⚙️ Run with docker

```shell
docker build -t promptmeteo:latest .
docker run --rm -i -t -v .:/home promptmeteo:latest
```

### ⚙️ Run tests

```
make test
```

## 📋 Current capacilities

### ✅ Available tasks

The current available tasks in Promptmeteo are:

| task_type            | description                       |
|----------------------|-----------------------------------|
| `DocumentQA`         | Document-level question answering |
| `DocumentClassifier` | Document-level classification     |
| `CodeGenerator`      | Code generation                   |
| `ApiGenerator`       | API REST generation               |
| `ApiFormatter`       | API REST correction               |
| `Summarizer`         | Text summarization                |

### ✅ Available Model

The current available `model_name` and `language` values are:

| model_provider | model_name           | language |
|----------------|----------------------|    ---   |
| openai         | gpt-3.5-turbo-16k    |    es    |
| openai         | gpt-3.5-turbo-16k    |    en    |
| azure          | gpt-3.5-turbo-16k    |    es    |
| azure          | gpt-3.5-turbo-16k    |    en    |
| hf_hub_api     | google/flan-t5-xxl   |    es    |
| hf_hub_api     | google/flan-t5-xxl   |    en    |
| hf_pipeline    | google/flan-t5-small |    es    |
| hf_pipeline    | google/flan-t5-small |    en    |
| google         | text-bison           |    es    |
| google         | text-bison           |    en    |
| google         | text-bison@001       |    es    |
| google         | text-bison@001       |    en    |
| google         | text-bison-32k       |    es    |
| google         | text-bison-32k       |    en    |
| bedrock        | anthropic.claude-v2  |    en    |
| bedrock        | anthropic.claude-v2  |    es    |
