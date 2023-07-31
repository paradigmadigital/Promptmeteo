---
annotations_creators:
- found
language_creators:
- found
language:
- de
- en
- es
- fr
- ja
- zh
license:
- other
multilinguality:
- monolingual
- multilingual
size_categories:
- 100K<n<1M
- 1M<n<10M
source_datasets:
- original
task_categories:
- summarization
- text-generation
- fill-mask
- text-classification
task_ids:
- text-scoring
- language-modeling
- masked-language-modeling
- sentiment-classification
- sentiment-scoring
- topic-classification
paperswithcode_id: null
pretty_name: The Multilingual Amazon Reviews Corpus
dataset_info:
- config_name: all_languages
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 364405048
    num_examples: 1200000
  - name: validation
    num_bytes: 9047533
    num_examples: 30000
  - name: test
    num_bytes: 9099141
    num_examples: 30000
  download_size: 640320386
  dataset_size: 382551722
- config_name: de
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 64485678
    num_examples: 200000
  - name: validation
    num_bytes: 1605727
    num_examples: 5000
  - name: test
    num_bytes: 1611044
    num_examples: 5000
  download_size: 94802490
  dataset_size: 67702449
- config_name: en
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 58601089
    num_examples: 200000
  - name: validation
    num_bytes: 1474672
    num_examples: 5000
  - name: test
    num_bytes: 1460565
    num_examples: 5000
  download_size: 86094112
  dataset_size: 61536326
- config_name: es
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 52375658
    num_examples: 200000
  - name: validation
    num_bytes: 1303958
    num_examples: 5000
  - name: test
    num_bytes: 1312347
    num_examples: 5000
  download_size: 81345461
  dataset_size: 54991963
- config_name: fr
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 54593565
    num_examples: 200000
  - name: validation
    num_bytes: 1340763
    num_examples: 5000
  - name: test
    num_bytes: 1364510
    num_examples: 5000
  download_size: 85917293
  dataset_size: 57298838
- config_name: ja
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 82401390
    num_examples: 200000
  - name: validation
    num_bytes: 2035391
    num_examples: 5000
  - name: test
    num_bytes: 2048048
    num_examples: 5000
  download_size: 177773783
  dataset_size: 86484829
- config_name: zh
  features:
  - name: review_id
    dtype: string
  - name: product_id
    dtype: string
  - name: reviewer_id
    dtype: string
  - name: stars
    dtype: int32
  - name: review_body
    dtype: string
  - name: review_title
    dtype: string
  - name: language
    dtype: string
  - name: product_category
    dtype: string
  splits:
  - name: train
    num_bytes: 51947668
    num_examples: 200000
  - name: validation
    num_bytes: 1287106
    num_examples: 5000
  - name: test
    num_bytes: 1302711
    num_examples: 5000
  download_size: 114387247
  dataset_size: 54537485
config_names:
- all_languages
- de
- en
- es
- fr
- ja
- zh
---

# Dataset Card for The Multilingual Amazon Reviews Corpus

## Table of Contents
- [Dataset Card for amazon_reviews_multi](#dataset-card-for-amazon_reviews_multi)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
      - [plain_text](#plain_text)
    - [Data Fields](#data-fields)
      - [plain_text](#plain_text-1)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Webpage:** https://registry.opendata.aws/amazon-reviews-ml/
- **Paper:** https://arxiv.org/abs/2010.02573
- **Point of Contact:** [multilingual-reviews-dataset@amazon.com](mailto:multilingual-reviews-dataset@amazon.com)

### Dataset Summary

We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish.

## Dataset Structure

### Data Instances

Each data instance corresponds to a review. The original JSON for an instance looks like so (German example):

```json
{
    "review_id": "de_0784695",
    "product_id": "product_de_0572654",
    "reviewer_id": "reviewer_de_0645436",
    "stars": "1",
    "review_body": "Leider, leider nach einmal waschen ausgeblichen . Es sieht super h\u00fcbsch aus , nur leider stinkt es ganz schrecklich und ein Waschgang in der Maschine ist notwendig ! Nach einem mal waschen sah es aus als w\u00e4re es 10 Jahre alt und hatte 1000 e von Waschg\u00e4ngen hinter sich :( echt schade !",
    "review_title": "Leider nicht zu empfehlen",
    "language": "de",
    "product_category": "home"
}
```

### Data Fields

- `review_id`: A string identifier of the review.
- `product_id`: A string identifier of the product being reviewed.
- `reviewer_id`: A string identifier of the reviewer.
- `stars`: An int between 1-5 indicating the number of stars.
- `review_body`: The text body of the review.
- `review_title`: The text title of the review.
- `language`: The string identifier of the review language.
- `product_category`: String representation of the product's category.

### Data Splits

Each language configuration comes with its own `train`, `validation`, and `test` splits. The `all_languages` split
is simply a concatenation of the corresponding split across all languages. That is, the `train` split for
`all_languages` is a concatenation of the `train` splits for each of the languages and likewise for `validation` and
`test`.

## Dataset Creation

### Curation Rationale

The dataset is motivated by the desire to advance sentiment analysis and text classification in other (non-English)
languages.

### Source Data

#### Initial Data Collection and Normalization

The authors gathered the reviews from the marketplaces in the US, Japan, Germany, France, Spain, and China for the
English, Japanese, German, French, Spanish, and Chinese languages, respectively. They then ensured the correct
language by applying a language detection algorithm, only retaining those of the target language. In a random sample
of the resulting reviews, the authors observed a small percentage of target languages that were incorrectly filtered
out and a very few mismatched languages that were incorrectly retained.

#### Who are the source language producers?

The original text comes from Amazon customers reviewing products on the marketplace across a variety of product
categories.

### Annotations

#### Annotation process

Each of the fields included are submitted by the user with the review or otherwise associated with the review. No
manual or machine-driven annotation was necessary.

#### Who are the annotators?

N/A

### Personal and Sensitive Information

According to the original dataset [license terms](https://docs.opendata.aws/amazon-reviews-ml/license.txt), you may not:
- link or associate content in the Reviews Corpus with any personal information (including Amazon customer accounts), or 
- attempt to determine the identity of the author of any content in the Reviews Corpus.

If you violate any of the foregoing conditions, your license to access and use the Reviews Corpus will automatically
terminate without prejudice to any of the other rights or remedies Amazon may have.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset is part of an effort to encourage text classification research in languages other than English. Such
work increases the accessibility of natural language technology to more regions and cultures. Unfortunately, each of
the languages included here is relatively high resource and well studied.

### Discussion of Biases

The dataset contains only reviews from verified purchases (as described in the paper, section 2.1), and the reviews
should conform the [Amazon Community Guidelines](https://www.amazon.com/gp/help/customer/display.html?nodeId=GLHXEX85MENUE4XF).

### Other Known Limitations

The dataset is constructed so that the distribution of star ratings is balanced. This feature has some advantages for
purposes of classification, but some types of language may be over or underrepresented relative to the original
distribution of reviews to achieve this balance.

## Additional Information

### Dataset Curators

Published by Phillip Keung, Yichao Lu, György Szarvas, and Noah A. Smith. Managed by Amazon.

### Licensing Information

Amazon has licensed this dataset under its own agreement for non-commercial research usage only. This licence is quite restrictive preventing use anywhere a fee is received including paid for internships etc.  A copy of the agreement can be found at the dataset webpage here:
https://docs.opendata.aws/amazon-reviews-ml/license.txt

By accessing the Multilingual Amazon Reviews Corpus ("Reviews Corpus"), you agree that the Reviews Corpus is an Amazon Service subject to the [Amazon.com Conditions of Use](https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&nodeId=508088) and you agree to be bound by them, with the following additional conditions:

In addition to the license rights granted under the Conditions of Use, Amazon or its content providers grant you a limited, non-exclusive, non-transferable, non-sublicensable, revocable license to access and use the Reviews Corpus for purposes of academic research. You may not resell, republish, or make any commercial use of the Reviews Corpus or its contents, including use of the Reviews Corpus for commercial research, such as research related to a funding or consultancy contract, internship, or other relationship in which the results are provided for a fee or delivered to a for-profit organization. You may not (a) link or associate content in the Reviews Corpus with any personal information (including Amazon customer accounts), or (b) attempt to determine the identity of the author of any content in the Reviews Corpus. If you violate any of the foregoing conditions, your license to access and use the Reviews Corpus will automatically terminate without prejudice to any of the other rights or remedies Amazon may have.

### Citation Information

Please cite the following paper (arXiv) if you found this dataset useful:

Phillip Keung, Yichao Lu, György Szarvas and Noah A. Smith. “The Multilingual Amazon Reviews Corpus.” In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020.

```
@inproceedings{marc_reviews,
    title={The Multilingual Amazon Reviews Corpus},
    author={Keung, Phillip and Lu, Yichao and Szarvas, György and Smith, Noah A.},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
    year={2020}
}
```

### Contributions

Thanks to [@joeddav](https://github.com/joeddav) for adding this dataset.
