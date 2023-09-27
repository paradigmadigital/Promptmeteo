![python-versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![code-lint](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/code_lint.yml/badge.svg)
![code-testing](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/code_testing.yml/badge.svg)
![publish-docker](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/publish_docker.yml/badge.svg)
![publish-pypi](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/publish_package.yml/badge.svg)
[![codecov](https://codecov.io/gh/paradigmadigital/Promptmeteo/branch/main/graph/badge.svg?token=KFJS6BGFH8)](https://codecov.io/gh/DelgadoPanadero/PromptMeteo)

# PromptMeteo 🔥🧔

**Promptmeteo** is a Promt Engineer Python library build over LangChain that simplfies the use of LLMs for different tasks with a low-code interface. For doing so, Promptmeteo is able to use different LLMs model and mcreate the prompts dinamically for concrete task underneath, given just some configuration parameters.




&nbsp;

# 🤔 What is this for?
**TL;DR: Industrialize projects powered by LLMs easily.**

LLMs are able to solve many tasks given a concrete instruction as input (prompt) and they can be used as a "reasoning engine" to build applications. However, this applications are very difficult to deploy and insdustrialize because two reasons. First because prompts usually include application logic in their definition, however, they are actually treated as a input argument. The latter mean that a bad prompt input can break the application.

Secondly, writing the concrete prompt for each task is not only a tedious work but also difficult. Slights changes in the input prompt can become in different results which make them very error-prone. Moreover when writting the prompt we do not only should take into consideration the task, but also the LLM that is going to use it, the model capacity...




&nbsp;
# 🚀 How do we solve it?

**TL;DR: Treating prompts an code equally!!!**

Prompmeteo try to solve the problems mentioned before by sepaarating the prompt definition in two parts: the task-logic (which is coded in prompt templates) and the concrete-problem (which is included as argument variables). Prompmeteo include high level objects for different tasks which are programmed with `.py` and `.prompt` files.

#### 🏠 Prebuilt tasks

The project includes high-level objects to solve different NLP tasks such as: text classification, Named Entity Recognition, code generation... This object only require configuration parameters to run and return the expected output from the task (i.e. we do not require to parse the output from the LLM).

#### ⚙️ Ease Deployment

The modules from Promptmeteo follow a similar model interface as a Scikit-Learn. Defining a interface with independant methods for training and predicting as well as saving and loading the model, allows Promptmeteo to be trained in and independant pipeline from predicting. This allows to reuse the conventional ML pipeline for LLM projects. 

#### 📦 Model Artifacts

LLMs can improve the results by including examples in their prompt. Promptmeteo is able to be trained with examples and to ensure reproducibility, the training process can be stored as a binary model artifact. This allows to store the training results and reuse it many times in new data. The training process store the embeddings from the input text in a vector database such as FAISS.

#### ⚙️ LLMs integration 

Prompmeteo include the integration of different LLMs throught LangChain. This includes models that can be executed locally as well as remote API calls from providers such as OpenAI and HuggingFace. 

#### 📄 Prompt Templating

Defining a concrete format when creating the prompts in Promptmeteo (`.prompt`), does not only allow to use it easily in a programatic way, but it also allows to versionate the prompts, understand where is the change when something happends and also **define code test oriented to prompt testing**. This testing includes aspects such as: validate the use of the language, that the size of the prompt is appropiate for the model,...

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

### ✨ Create the task
You can make a prediccion directly indanciating the model and calling the method `predict()`.

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
Buy you can also include examples to improve the results by calling the method `train()`


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

One the model is trained it can be save locally

```python
clf.save_model("hello_world.prompt")
```


### ✨ Load a trained task

and loaded again to make new predictions

```python
from promptmeteo import DocumentClassifier

clf = DocumentClassifier(
        language            = 'en',
        model_provider_name = 'hf_pipeline',
        model_name          = 'google/flan-t5-small',
    ).load_model("hello_world.prompt")

clf.predict(['so cool!!'])
```

```shell
[['positive']]
```

###  ✨ Learn more

More examples can be seen in the directory [examples](./examples).

&nbsp;


## 🚗 Usage

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

|       task_type       |            description             |
|          ---          |                ---                 |
|      `DocumentQA`     |  Document-level question answering |
|  `DocumentClassifier` |    Document-level classification   |
|     `CodeGenerator`   |          Code generation           |

### ✅ Available Model

The current available `model_name` and `language` values are:

| model_provider |       model_name          | language |
|      ---       |           ---             |    ---   |
|     openai     |     text-davinci-003      |    es    |
|     openai     |     text-davinci-003      |    en    |
|   hf_hub_api   |    google/flan-t5-xxl     |    es    |
|   hf_hub_api   |    google/flan-t5-xxl     |    en    |
|  hf_pipeline   |   google/flan-t5-small    |    es    |
|  hf_pipeline   |   google/flan-t5-small    |    en    |
|  google        |   text-bison              |    en    |
|  google        |   text-bison@001          |    en    |
|  google        |   text-bison-32k          |    en    |
