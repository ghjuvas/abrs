'''
Module for preprocessing data before classification.
'''
import stanza
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class NLP:

    def __init__(self) -> None:
        self.nlp = stanza.Pipeline('ru', processors='tokenize', verbose=False)

    def preprocess(self, text) -> list:
        '''
        Get sentences and tokens from text.
        '''
        doc = self.nlp(text)
        sentences = []
        tokens_starts = []
        tokens_ends = []
        for sent in doc.sentences:
            sent_tokens = []
            sent_starts = []
            sent_ends = []
            for token in sent.tokens:
                sent_tokens.append(token.text)
                sent_starts.append(token.start_char)
                sent_ends.append(token.end_char)
            sentences.append(sent_tokens)
            tokens_starts.append(sent_starts)
            tokens_ends.append(sent_ends)

        return sentences, tokens_starts, tokens_ends


class Normalizer:

    def __init__(self, model: str, tokenizer: str) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer)

    def normalize(self, x, **kwargs):
        inputs = self.tokenizer(x, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
