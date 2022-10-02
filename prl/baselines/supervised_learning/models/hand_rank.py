# hand strength pseudo code should be adjusted to multiple oppcards

def rank(hand, opp_cards, board):
    """Allows for empty board"""

    """
    I can maybe do the following:
    
    look at preflop heads up equities of 1326 hands
    look at the drop in equity when going from heads up to three players (equity-delta)
    given a starting hand, increase the required equity needed to fold inverse to the equity-delta
    e.g. AA had 80 % in heads up and 70% in three player games
    increase the fold threshold from 50% to (50+(80-70))=60% 
    I would have to run 1326 * len(2,3,4,5,6) = 1326 * 5 =  6630 Monte-carlo simulations
    to create a 1326x5 lookup table
    
    that is for preflop - all good
    now for postflop, we can compute HandStrenght given the board and only ONE OPPONENT, and APPLY THE SAME 
    equity delta: lets say our AA equity decreases by 20 percent on the flop, to 60% HandStrenght
    then we apply our delta from the lookup table, i.e. we add 10% to the decrease,
    resulting in an Adjusted HandStrength of 50% which is on the edge of folding.
    
    todo: how to mathematically justify that heuristic, that if equity drops with increasing 
    number of players with an unknown board, it will drop with approximately the same rate
    for a known board 
    
    
    online: flop gegeben; 2 spieler; ehs wie im paper + abzug von equity delta aus lut
offline: preflop state; aber monte carlo mit variabler spielerzahl -> lut: 1326 x N,  lut_ij in [0,1] gibt das equity delta

online: flop steht fest
offline: flop wird auch im monte carlo mitgewuerfelt

i) monte carlo mit N spielern preflop start
ii) deltas wegschreiben
iii) ehs wie im paper fuer 2 spieler mit gegebenen flop
iv) delta auslesen fuer start hand index und von ehs abziehen
"""