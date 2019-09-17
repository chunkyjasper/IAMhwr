from nltk.lm.api import LanguageModel
from nltk.lm.smoothing import KneserNey


# KneserNey of NLTK, but implemented with stupid backoff
class KneserNeyBackoff(LanguageModel):
    def __init__(self, order, backoff=0.4, **kwargs):
        super().__init__(order, **kwargs)
        self.backoff = backoff
        self.estimator = KneserNey(self.vocab, self.counts)

    def unmasked_score(self, word, context=None):
        if not context:
            return self.estimator.unigram_score(word)
        if not self.counts[context][word]:
            return self.backoff * self.unmasked_score(word, context[1:])
        alpha, gamma = self.estimator.alpha_gamma(word, context)
        return alpha + gamma * self.unmasked_score(word, context[1:])