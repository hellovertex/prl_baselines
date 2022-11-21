#include <iostream>
#include <random>
#include <string>
#include <iterator>
#include <algorithm>
#include "ext/src/SevenEval.h"
#include <pybind11/pybind11.h>



namespace py = pybind11;
const int n_remaining_deck_cards = 47;
int run_mc(int c0, int c1,  // player cards
           int b0, int b1, int b2, int b3, int b4, // board cards or -1
           int n_opponents, // number of opponents
           int n_iters) {  // number of iterations to run mc
    // Get the rank of the seven-card spade flush, ace high.
    // std::cout << SevenEval::GetRank(1, 41, 18, 19, 16, 20, 24)  << std::endl;

    // BUILD REMAINING DECK - REMOVE DRAWN CARDS FROM DECK
    std::array<int, n_remaining_deck_cards> deck;
    std::size_t idx = 0;
    for (std::size_t i=0; i<n_remaining_deck_cards; i++) {
        if (i!=c0 && i!=c1 && i!=b0 &&i!=b1 && i!=b2 && i!=b3 && i!=b4){
            deck[idx] = i;
            idx++;
        }
    }
    // SAMPLE MISSING CARDS FROM REMAINING DECK WITHOUT REPLACEMENT
    int cards_to_draw = n_opponents * 2;
    if(b0 == -1) {
        // draw flop + turn + river
        std::vector<int>::iterator drawn;
        std::sample(deck.begin(),deck.end(), drawn, cards_to_draw + 5,
                    std::mt19937{std::random_device{}()});
    } else if (b3 == -1 && b4 == -1) {
        // draw turn + river
    } else if (b4==-1){
        // draw river
    }

    // DEAL OPPONENT CARDS
    for(std::size_t j = 0; j<n_opponents; j++) {

    }
    return SevenEval::GetRank(c0, c1, b0, b1, b2, b3, b4);
}

PYBIND11_MODULE(hand_eval, handle) {
    handle.doc() = "This function runs monte carlo simulation to determine a 7-card rank.";
    handle.def("run_monte_carlo", &run_mc);
}