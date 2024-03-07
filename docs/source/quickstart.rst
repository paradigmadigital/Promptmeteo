⚡ Quickstart
================

.. _quickstart:

✨ Create the task
---------------------

You can make a prediction directly indicating the model and calling the method `predict()`.

.. code-block:: python

   from promptmeteo import DocumentClassifier

    clf = DocumentClassifier(
            language            = 'en',
            model_provider_name = 'hf_pipeline',
            model_name          = 'google/flan-t5-small',
            prompt_labels       = ['positive', 'neutral', 'negative']
        )

    clf.predict(['so cool!!'])

.. code-block:: console

    [['positive']]

✨ Train the task
-------------------------

You can also include examples to improve the results by calling the method `train()`

.. code-block:: python

    clf = clf.train(
        examples    = ['i am happy', 'doesnt matter', 'I hate it'],
        annotations = ['positive', 'neutral', 'negative'],
    )

    clf.predict(['so cool!!'])

.. code-block:: console

    [['positive']]

✨ Save a trained task
-------------------------

Once the model is trained it can be saved locally

.. code-block:: console

    clf.save_model("hello_world.meteo")

✨ Load a trained task
-------------------------

and loaded again to make new predictions

.. code-block:: python

    from promptmeteo import DocumentClassifier

    clf = DocumentClassifier(
            language            = 'en',
            model_provider_name = 'hf_pipeline',
            model_name          = 'google/flan-t5-small',
        ).load_model("hello_world.meteo")

    clf.predict(['so cool!!'])


.. code-block:: console

    [['positive']]


Models can also be loaded without instantiating the class by using load_model as a function instead of a method:

.. code-block:: python

    from promptmeteo import DocumentClassifier

    clf = DocumentClassifier.load_model("hello_world.meteo")