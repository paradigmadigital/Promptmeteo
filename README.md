![python-versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![code-lint](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/code_lint.yml/badge.svg)
![code-testing](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/code_testing.yml/badge.svg)
![publish-docker](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/publish_docker.yml/badge.svg)
![publish-pypi](https://github.com/paradigmadigital/Promptmeteo/actions/workflows/publish_package.yml/badge.svg)
[![codecov](https://codecov.io/gh/paradigmadigital/Promptmeteo/branch/main/graph/badge.svg?token=KFJS6BGFH8)](https://codecov.io/gh/DelgadoPanadero/PromptMeteo)

# PromptMeteo 🔥🧔

**Promptmeteo** is a Python library for prompt engineering built over LangChain. It simplifies the utilization of large language models (LLMs) for various tasks through a low-code interface. To achieve this, Promptmeteo can employ different LLM models and dynamically generate prompts for specific tasks based on just a few configuration parameters.



# 🤔 What is this library for?
**TL;DR: Industrialize projects powered by LLMs easily.**

LLMs have the capability to address various tasks when provided with specific instructions in the form of input prompts. They can function as a "reasoning engine" for constructing applications. However, deploying and industrializing these applications poses significant challenges for two main reasons.

Firstly, prompts typically encapsulate application logic in their definition, yet they are treated merely as input arguments. This means that a poorly formulated prompt input has the potential to disrupt the application.

Secondly, crafting concrete prompts for each task is not only a laborious task but also a complex one. Minor alterations in the input prompt can result in different outcomes, rendering them highly error-prone. Additionally, when composing prompts, considerations extend beyond the task itself to include factors such as the specific LLM being used, the model's capacity, and other relevant aspects.


# ⚡ Use

Find all tutorials, a quick start and much more information in [our official documentation](https://paradigmadigital.github.io/promptmeteo-docs/>).

# Contribute

We encourage you:

* to [open issues](https://github.com/paradigmadigital/Promptmeteo/issues/)
* to [contact us](https://github.com/paradigmadigital/Promptmeteo/graphs/contributors>) to talk about the library
* to [develop open issues or fix bugs](https://github.com/paradigmadigital/Promptmeteo/issues?q=is%3Aissue+is%3Aopen+>)

We are still working in the contribution guidelines and code of conduct. Be **respectful and kind, and remember that anybody makes mistakes**.

## Develop

Clone the repository and create a new branch based on [semantic messages](https://www.conventionalcommits.org/en/v1.0.0/#summary>) (_feat/feature_, _chore_, etc):

```bash
git clone git@github.com:paradigmadigital/Promptmeteo.git
cd promptmeteo
git checkout -b branch
```

Install the development dependencies:

```bash
make dev
```

Pre-commit hooks are installed after setting the project in which the format, lint and tests are passed through the project. Passing all the hooks is the minimum requirements to push to the project.

## Documentation

To install documentation dependencies and build the documentation download the repository and:

```bash
make docsetup
make html
```

A lot of new files will be automatically generated, **be careful to only add the strictly necessary**.