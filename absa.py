'''
Module for performing sentiment analysis using transformer model.
'''
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class SentimentAnalyzer:

    def __init__(self, model: str, tokenizer: str) -> None:
        self.BIO = ['B-POS', 'I-POS',
                    'B-NEG', 'I-NEG',
                    'B-NEUT', 'I-NEUT', 'O']
        self.label2id = {label: i for i, label in enumerate(self.BIO)}
        self.id2label = {i: label for i, label in enumerate(self.BIO)}
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # sample ner inference
    def get_sentiment(self, data: list):
        '''
        Get sentiment labels using sequence labeling model.
        '''
        pred_labels = []
        pred_aspects = []
        pred_embeddings = []
        pred_sentiment = []

        for sent in data:
            encodings = self.tokenizer(sent, truncation=True,
                                       padding=True,
                                       is_split_into_words=True)
            inputs = self.tokenizer.encode(sent, truncation=True,
                                           padding=True,
                                           is_split_into_words=True,
                                           return_tensors="pt")

            # get model output
            output = self.model(inputs, output_hidden_states=True)
            outputs = output[0]
            embeddings = output.hidden_states[0][0]
            preds = torch.argmax(outputs, dim=2)[0].tolist()

            # align predictions
            aligned_preds = []
            aligned_embs = []
            word_ids = encodings.word_ids()
            previous_word_idx = None
            aligned_embedding = np.zeros(768)
            aligned_num = 0
            for idx, word_idx in enumerate(word_ids):
                if word_idx is not None:
                    aligned_embedding += embeddings[idx].detach().numpy()
                    aligned_num += 1
                    if word_idx != previous_word_idx:
                        previous_word_idx = word_idx
                        aligned_preds.append(preds[idx])

                        aligned_embedding /= aligned_num
                        aligned_embs.append(aligned_embedding)

                        aligned_embedding = np.zeros(768)
                        aligned_num = 0

            assert len(aligned_preds) == len(aligned_embs)

            # get sentiment, aspects ids and embeddings of aspects
            sent_aspects = []
            sent_embs = []
            sent_labels = []
            sent_sentiment = []

            cur_embedding = np.zeros(768)
            cur_num = 0
            cur_start = None
            cur_end = None
            cur_sent = None
            for idx, (sent_idx, sent_emb) in enumerate(zip(aligned_preds, aligned_embs)):
                sent = self.id2label.get(sent_idx, None)
                cur_sent = sent
                sent_labels.append(sent)

                # not O
                if sent != 'O':
                    if sent.startswith('B'):
                        # if B but previous was I
                        if cur_start is not None:
                            cur_embedding /= cur_num
                            sent_embs.append(cur_embedding)

                            sent_aspects.append((cur_start, cur_end))
                            if 'NEUT' in cur_sent:
                                sent_sentiment.append('neutral')
                            elif 'NEG' in cur_sent:
                                sent_sentiment.append('negative')
                            else:
                                sent_sentiment.append('positive')

                            cur_embedding = np.zeros(768)
                            cur_num = 0
                            cur_start = None
                            cur_end = None

                    # I and B
                    cur_embedding += sent_emb
                    # print(cur_embedding.shape)
                    cur_num += 1
                    if cur_start is None:
                        cur_start = idx
                    cur_end = idx

                # O
                else:
                    if cur_embedding is not None and \
                        cur_num is not None and \
                        cur_start is not None and \
                        cur_end is not None:
                        cur_embedding /= cur_num
                        sent_embs.append(cur_embedding)

                        sent_aspects.append((cur_start, cur_end))
                        if 'NEUT' in cur_sent:
                            sent_sentiment.append('neutral')
                        elif 'NEG' in cur_sent:
                            sent_sentiment.append('negative')
                        else:
                            sent_sentiment.append('positive')

                        cur_embedding = np.zeros(768)
                        cur_num = 0
                        cur_start = None
                        cur_end = 0

            pred_labels.append(sent_labels)
            pred_embeddings.append(sent_embs)
            pred_aspects.append(sent_aspects)
            pred_sentiment.append(sent_sentiment)

        return pred_labels, pred_aspects, pred_embeddings, pred_sentiment
