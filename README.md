# Aspect-based review summarization
Program code for aspect-based review summarization

## About
This program performs aspect-based sentiment analysis using fine-tuned XLM-RoBERTa and summarization using clustering algorithm Affinity Propagation.

## Technical
* Python3
* ABSA: `torch`, `transformers`
* Summarization: `scikit-learn`, `scipy`
* Normalization: `torch`, `transformers`

## Requirements
You need Python3 environment. To install needed libraries, write in terminal from current folder this command:
> `pip3 install -r requirements.txt`

## Models
To perform ABSA you need model from this link:
> [XLM-RoBERTa](https://drive.google.com/drive/folders/1GGQIdoVfgbuSaNuDuyOH1wFZwQpxqKYH?usp=sharing)

To perform normalization you need model from this link:
> [ruT5](https://drive.google.com/drive/folders/1oW9vCsMgkW1JMIKzPiFG29CG_jBjAWg0?usp=sharing)

## Sample
To run code write following lines in terminal:
> `python3 summarization.py corpus.txt` -- to perform summarization
> `python3 summarization.py -n corpus.txt` -- to add normalization before printing

## About me
Yanina Khudina, currently enrolled in bachelor program "Fundamential and Computational Linguistics" at HSE University, Moscow.
* e-mail: yykhudina@yandex.ru
