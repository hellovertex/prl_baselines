class Vectorizer:
    """ Abstract Vectorizer Interface. All vectorizers should be derived from this base class
    and implement the method "vectorize"."""

    def vectorize(self, obs, *args, **kwargs):
        """todo"""
        raise NotImplementedError