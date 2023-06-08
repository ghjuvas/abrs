'''
Module for get result using data.
'''
from argparse import ArgumentParser
from collections import defaultdict
import logging
import os
from nlp import NLP, Normalizer
from absa import SentimentAnalyzer
from summarize import AffinityPropagationClusterisator

logging.basicConfig(level=logging.INFO)

SENTIMENT_MODEL = 'xlmroberta-sentiment-seqlabeling_both'
SENTIMENT_TOKENIZER = 'xlm-roberta-base'


class CustomArgumentParser(ArgumentParser):

    def __init__(self) -> None:
        super().__init__()
        self.add_argument('path',
                          type=str,
                          help='Path to texts. Must be separated using blank line')
        self.add_argument('-n', '--normalization',
                          action='store_true',
                          help='Normalize aspects before printing')


class Summarization:

    def __init__(self) -> None:

        self.nlp = NLP()
        self.absa = SentimentAnalyzer(
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_TOKENIZER
        )
        self.summarize = AffinityPropagationClusterisator()
        self.path = None
        self.corpus = None
        self.normalization = False

    def _get_texts(self, path):
        '''
        Get texts from path.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')
        if len(texts) <= 1:
            raise ValueError('Too small corpus for summarization!')

        return texts

    def get_summarization(self, path, normalization=False):
        logging.info('Getting texts...')
        texts = self._get_texts(path)

        sentiment = []
        embeddings = []
        mentions = []
        logging.info('Getting mentions and sentiment...')
        sum_aspects = 0
        for text in texts:
            preprocessed = self.nlp.preprocess(text)
            sentiment_output = self.absa.get_sentiment(preprocessed[0])
            for sent_idx, sent in enumerate(sentiment_output[1]):  # aspects ids
                for aspect_idx, aspect in enumerate(sent):
                    sum_aspects += 1
                    start = preprocessed[1][sent_idx][aspect[0]]
                    end = preprocessed[2][sent_idx][aspect[1]]
                    mention = text[start:end]
                    # print(mention)
                    sentiment.append(sentiment_output[3][sent_idx][aspect_idx])
                    embeddings.append(sentiment_output[2][sent_idx][aspect_idx])
                    mentions.append(mention)

        logging.info('Getting aspects...')
        sentiment_summarization = self.summarize.get_summarized_aspects(
            embeddings,
            mentions,
            sentiment)

        if normalization:
            logging.info('Getting normalized aspects...')
            norm = Normalizer(model='rut5-normalization',
                              tokenizer='cointegrated/rut5-base-multitask')

            new_sum = defaultdict(list)
            for sentiment, values in sentiment_summarization.items():
                for aspect in values:
                    normalized_aspect = norm.normalize(aspect)
                    new_sum[sentiment].append(normalized_aspect)

            sentiment_summarization = new_sum

        print('\n\n\n')
        print('SUMMARIZATION:')
        for sentiment, aspects in sentiment_summarization.items():
            print('Sentiment:', sentiment)
            print('Aspects:', aspects)
        print('\n\n\n')

        return aspects


if __name__ == '__main__':
    parser = CustomArgumentParser()
    args = parser.parse_args()
    path = args.path
    # handle extension errors
    if not os.path.exists(path):
        raise ValueError('Path is not valid!')
    if os.path.splitext(path)[1] not in ['.txt', '']:
        raise ValueError(
            'Program do not support this file extension! Please, use .txt or nothing'
            )
    summ = Summarization()
    summ.get_summarization(path, args.normalization)
