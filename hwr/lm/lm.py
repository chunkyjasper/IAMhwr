from nltk.lm.api import LanguageModel, Smoothing


class MLE(LanguageModel):
    def unmasked_score(self, word, context=None):
        context = context[-(self.order - 1):]
        return self.context_counts(context).freq(word)


class StupidBackoff(LanguageModel):
    def __init__(self, order, backoff=0.4, **kwargs):
        super().__init__(order=order, **kwargs)
        self.backoff = backoff

    def unmasked_score(self, word, context=None):
        if not context:
            return self.counts[1].freq(word)
        context = context[-(self.order - 1):]
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
        context = context[-(self.order - 1):]
        alpha, gamma = self.estimator.alpha_gamma(word, context)
        if alpha == 0 and gamma == 0:
            return 0
        return alpha + gamma * self.unmasked_score(word, context[1:])


# KneserNey of NLTK, but implemented with backoff
class KneserNeyBackoff(LanguageModel):
    def __init__(self, order, backoff=0.4, **kwargs):
        super().__init__(order, **kwargs)
        self.estimator = KneserNey(self.vocab, self.counts, backoff=backoff)

    def unmasked_score(self, word, context=None):
        if not context:
            return self.estimator.unigram_score(word)
        context = context[-(self.order - 1):]
        alpha, gamma = self.estimator.alpha_gamma(word, context)
        return alpha + gamma * self.unmasked_score(word, context[1:])


def _count_non_zero_vals(dictionary):
    return sum(1.0 for c in dictionary.values() if c > 0)


class KneserNey(Smoothing):

    def __init__(self, vocabulary, counter, discount=0.1, backoff=0.0, **kwargs):
        super(KneserNey, self).__init__(vocabulary, counter, *kwargs)
        self.discount = discount
        self.backoff = backoff

    def unigram_score(self, word):
        return 1.0 / len(self.vocab)

    def alpha_gamma(self, word, context):
        prefix_counts = self.counts[context]
        prefix_total_ngrams = prefix_counts.N()
        word_count_given_prefix = prefix_counts[word]
        if word_count_given_prefix:
            alpha = max(word_count_given_prefix - self.discount, 0.0) / prefix_total_ngrams
            gamma = (self.discount * _count_non_zero_vals(prefix_counts) / prefix_total_ngrams)
        else:
            alpha = 0.0
            gamma = self.backoff
        return alpha, gamma
