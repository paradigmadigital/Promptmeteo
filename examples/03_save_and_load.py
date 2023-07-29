from promptmeteo import Promptmeteo

# Instance Promptmeteo and train it
model = Promptmeteo(
    task_type           = 'classification',
    model_provider_name = 'hf_pipeline',
    model_name          = 'google/flan-t5-small',
    verbose             = True
).train(
    examples = ['estoy feliz', 'me da igual', 'no me gusta'],
    annotations = ['positivo', 'neutral', 'negativo'],
)


# Now you can save the training results
model.save_model('model.meteo')


# Finally load the model
model = Promptmeteo(
    task_type           = 'classification',
    model_provider_name = 'hf_pipeline',
    model_name          = 'google/flan-t5-small',
    verbose             = True
).load_model('model.meteo')

pred = model.predict(['que guay!!'])
