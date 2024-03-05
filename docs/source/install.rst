⚙️ Installation and configuration
=====

.. _install:

Install
------------

To install the stable version of the library use `pip`_.

.. _pip: https://pypi.org/

.. code-block:: console

   (.venv) $ pip install promptmeteo

Configure credentials
------------------------

Create a ```.env``` with the following variables depending on the LLM provider

Google Cloud
^^^^^^^^^^^^^^^^

First you should create a [Service Account](https://cloud.google.com/vertex-ai/docs/general/custom-service-account#configure) with the role: ``Vertex AI User.``

Once created, generate a key, store it locally and reference the path in the .env file:

.. code-block:: console

    GOOGLE_CLOUD_PROJECT_ID="MY_GOOGLE_LLM_PROJECT_ID"
    GOOGLE_APPLICATION_CREDENTIALS="PATH_TO_SERVICE_ACCOUNT_KEY_FILE.json"

OpenAI
^^^^^^^^^^^^

Create your Secret API key in your User settings [page](https://platform.openai.com/account/api-keys).

Indicate the value of the key in your .env file:

.. code-block:: console

    OPENAI_API_KEY="MY_OPENAI_API_KEY"


You can also pass `openai_api_key` as a named parameter.

Hugging Face
^^^^^^^^^^^^^^^^

Create Access Token in your User settings [page](https://huggingface.co/settings/tokens).

.. code-block:: console

    HUGGINGFACEHUB_API_TOKEN="MY_HF_API_KEY"

You can also pass `huggingfacehub_api_token` as a named parameter.

AWS Bedrock
^^^^^^^^^^^^^^^^

Create your access keys in security credentials of your user in AWS.

Then write in the files ```~/.aws/config``` and ````~/.aws/credentials```` for Linux and MacOS or ````%USERPROFILE%\.aws\config```` and ````%USERPROFILE%\.aws\credentials```` for Windows:

In credentials

.. code-block:: console

    [default]
    aws_access_key_id = <YOUR_CREATED_AWS_KEY>
    aws_secret_access_key = <YOUR_CREATED_AWS_SECRET_KEY>


In config:

.. code-block:: console

    [default]
    region = <AWS_REGION>

