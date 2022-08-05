class Writer:
    """ Abstract Writer Interface. All writers should be derived from this base class
    and implement the method `write_to_file`."""
    def write_train_data(self, *args, **kwargs):
        """Invokes a Parser to generate PokerEpisode instances.
        Then invokes an Encoder to generate training data with labels from PokerEpisodes.
        Finally writes these training_data and labels to disk.
        Args:
            abs_filepath: .txt file containing poker games,
                poker games are usually crawled from online platforms such as Pokerstars.
        """
        raise NotImplementedError
