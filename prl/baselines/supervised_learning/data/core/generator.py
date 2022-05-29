class TrainingDataGenerator:
    """ Abstract TrainingDataGenerator Interface. All generators should be derived from this base class
    and implement the method "generate_from_file"."""
    def generate_from_file(self, abs_filepath):
        """Invokes a Parser to generate PokerEpisode instances.
        Then invokes an Encoder to generate training data with labels from PokerEpisodes.
        Finally writes these training_data and labels to disk.
        Args:
            abs_filepath: .txt file containing poker games,
                poker games are usually crawled from online platforms such as Pokerstars.
        """
        raise NotImplementedError
