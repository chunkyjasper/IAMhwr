from nltk.lm.api import LanguageModel
from nltk.lm.smoothing import KneserNey


class StupidBackoff(LanguageModel):
    def __init__(self, order, backoff=0.4, **kwargs):
        super().__init__(order=order, **kwargs)
        self.backoff = backoff

    def unmasked_score(self, word, context=None):
        if not context:
            return self.counts[1].freq(word)
        context = context[:self.order]
        context_freq_d = self.context_counts(context)
        ngram_count = context_freq_d[word]
        if not ngram_count:
            return self.backoff * self.unmasked_score(word, context[1:])
        else:
            return ngram_count / context_freq_d.N()


# Implemention from NLTK. Override to fix zero division issue.
class KneserNeyInterpolated(LanguageModel):
    def __init__(self, order, **kwargs):
        super().__init__(order, **kwargs)
        self.estimator = KneserNey(self.vocab, self.counts)

    def unmasked_score(self, word, context=None):
        if not context:
            return self.estimator.unigram_score(word)
        if not self.counts[context]:
            # alpha = 0, gamma = 1
            return self.unmasked_score(word, context[1:])
        alpha, gamma = self.estimator.alpha_gamma(word, context)
        return alpha + gamma * self.unmasked_score(word, context[1:])
