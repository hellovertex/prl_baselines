from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns

# use vectorized representation of AugmentedObservationFeatureColumns to parse back to PokerEpisode object

def convert():
    # PokerEpisode.num_players -- positive stacks
    # PokerEpisode.blinds -- normalization_sum * sb, bb index

    pass