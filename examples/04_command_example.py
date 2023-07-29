import argparse

from promptmeteo import Promptmeteo


def main(args):

    with open(args.data_path) as fin:
        data = fin.read().split('\n')
        data = [row.split(';') for row in data]
        data = [row for row in data if len(row) == 2]

    with open(args.prompt_path) as fin:
        prompt = fin.read()

    cls = Promptmeteo(
        task_type='classification',
        model_provider_name='hf_pipeline',
        model_name='google/flan-t5-small',
        selector_algorithm='length',
        selector_k=3,
        verbose=True
    ).read_prompt_file(
        prompt
    )

    cls = cls.train(
        examples=[x for x, y in data],
        annotations=[y for x, y in data]
    )

    pred = cls.predict(['que guay!!'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
