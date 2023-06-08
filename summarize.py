'''
Module for summarization using clustering.
'''
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from copy import deepcopy
from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation


class Clusterisator(ABC):

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    @abstractmethod
    def _clusterisation(self):
        pass

    def _get_aspects(self, embeddings: list):
        '''
        Identify aspects after clusterisation.
        '''
        labels, centers = self._clusterisation(embeddings)

        labels = list(labels)

        min_dist_states = {}

        for idx, (label, embedding) in enumerate(zip(labels, embeddings)):
            if label not in min_dist_states:
                min_dist_states[label] = {
                    'min_dist': 1,
                    'min_idx': None,
                    'embedding': None}
            # get embedding with minimum cosine distance
            # from center of cluster
            dist = distance.cosine(embedding, centers[label])
            min_dist = min_dist_states[label].get('min_dist', None)
            if min_dist:
                if min_dist > dist:
                    if min_dist_states[label]['embedding'] is not None:
                        # return previous embedding
                        # with minimum distance into the list
                        emb_idx = min_dist_states[label]['min_idx']
                        emb = min_dist_states[label]['embedding']
                        embeddings[emb_idx] = emb

                    min_dist_states[label]['min_dist'] = dist
                    min_dist_states[label]['min_idx'] = idx
                    min_dist_states[label]['embedding'] = embedding

                    # remove main aspect and it's embedding from clusters
                    # leave other embeddings of clusters
                    # to get summarized polarity
                    embeddings[idx] = None

        return labels, embeddings

    def get_summarized_aspects(self, embeddings: list,
                               mentions: list, sentiment: list):
        '''
        Summarize sentiment by aspect in the clusters.
        '''
        embeddings = deepcopy(embeddings)
        labels, embeddings = self._get_aspects(embeddings)

        aspects = [0 for _ in range(len(set(labels)))]
        other_sentiment = [Counter() for _ in range(len(set(labels)))]

        for idx, (label, embedding) in enumerate(zip(labels, embeddings)):

            # get main aspect
            if embedding is None:
                aspects[label] = mentions[idx]
            other_sentiment[label][sentiment[idx]] += 1

        sentiments = [aspect.most_common()[0][0] for aspect in other_sentiment]

        sentiment_summarization = defaultdict(list)
        for asp, sent in zip(aspects, sentiments):
            sentiment_summarization[sent].append(asp)

        return sentiment_summarization


class AffinityPropagationClusterisator(Clusterisator):

    def _clusterisation(self, embeddings: list) -> tuple:

        affp = AffinityPropagation(random_state=self.random_state, damping=0.7)
        fitted = affp.fit(embeddings)
        labels = fitted.labels_
        centers = fitted.cluster_centers_

        return labels, centers
