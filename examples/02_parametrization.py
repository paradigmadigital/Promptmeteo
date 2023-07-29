from promptmeteo import Promptmeteo

# To create the prompt instrucction you have to use a string with a
# structure like this one

prompt = '''
TEMPLATE:
    "
    Clasifica el siguiente texto en una de las siguientes clases:
    {__LABELS__}.

    {__TASK_INFO__}

    {__ANSWER_FORMAT__}

    {__CHAIN_OF_THOUGHTS__}
    "

LABELS:
    ["positive", "negative", "neutral"]

TASK_INFO:
    "Asume que estamos usando estas categorías con su significado subjetivo
    habitual: algo positivo puede describirse como bueno, de buena calidad,
    deseable, útil y satisfactorio; algo negativo puede describirse como
    malo, de mala calidad, indeseable, inútil o insatisfactorio; neutro es
    la clase que asignaremos a todo lo que no sea positivo o negativo."

ANSWER_FORMAT:
    "En tu respuesta incluye sólo el nombre de la clase, como una única
    palabra ({__LABELS__}), en minúscula, sin puntuación, y sin añadir
    ninguna otra afirmación o palabra."

CHAIN_OF_THOUGHTS:
    "Por favor argumenta tu respuesta paso a paso, explica por qué crees que
    está justificada tu elección final, y por favor asegúrate de que acabas
    tu explicación con el nombre de la clase que has escogido como la
    correcta, en minúscula y sin puntuación."
'''

model = Promptmeteo(
    task_type='classification',
    model_provider_name='hf_pipeline',
    model_name='google/flan-t5-small',
).read_prompt_file(prompt)


# You can also make it through argument variables
prompt_labels=['positive','negative','neutral'],

prompt_task_info="""
Asume que estamos usando estas categorías con su significado
subjetivo habitual: algo positivo puede describirse como bueno,
de buena calidad, deseable, útil y satisfactorio; algo negativo
puede describirse como malo, de mala calidad, indeseable, inútil
o insatisfactorio; neutro es la clase que asignaremos a todo lo
que no sea positivo o negativo.""",

prompt_answer_format="""
En tu respuesta incluye sólo el nombre de la clase, como una única
palabra ({__LABELS__}), en minúscula, sin puntuación, y sin añadir
ninguna otra afirmación o palabra.""",

prompt_chain_of_thoughts="""
Por favor argumenta tu respuesta paso a paso, explica por qué crees
que está justificada tu elección final, y por favor asegúrate de
que acabas tu explicación con el nombre de la clase que has elegido
como la correcta, en minúscula y sin puntuación.""",

model = Promptmeteo(
    task_type='classification',
    model_provider_name='hf_pipeline',
    model_name='google/flan-t5-small',
    prompt_labels=prompt_labels,
    prompt_task_info=prompt_task_info,
    prompt_answer_format=prompt_answer_format,
    prompt_chain_of_thoughts=prompt_chain_of_thoughts)
