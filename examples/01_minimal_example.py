from promptmeteo import Promptmeteo

# Load Promptmeteo with the minimal configuration for a classification
# task
model = Promptmeteo(
    task_type           = 'classification',
    model_provider_name = 'hf_pipeline',
    model_name          = 'google/flan-t5-small',
)

# Train prompmeteo with some examples and its labels (examples and annotations
# must be a list of strings)
model = model.train(
    examples = ['estoy feliz', 'me da igual', 'no me gusta'],
    annotations = ['positivo', 'neutral', 'negativo'],
)

# Predict a new samples (the output should be a list of strings with the
# result for each sample)
pred = model.predict(['que guay!!'])
