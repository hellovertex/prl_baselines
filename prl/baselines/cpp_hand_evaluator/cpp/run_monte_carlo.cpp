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
    int cards_to_sample = n_opponents * 2; // board cards will be added below
    bool draw_after_preflop = false;
    bool draw_turn_and_river = false;
    bool draw_only_river = false;

    // to store board of current simulation round in case of missing board cards
    int cur_b0;
    int cur_b1;
    int cur_b2;
    int cur_b3;
    int cur_b4;

    // Initialize Monte Carlo Simulation
    if(b0 == -1 && b3 == -1 && b4 == -1) {
        // draw flop + turn + river
        cards_to_sample += 5;
        draw_after_preflop = true;
    }
    else if (b3 == -1 && b4 == -1) {
        // draw turn + river
        cards_to_sample += 2;
        draw_turn_and_river = true;
    }
    else if (b4==-1){
        // draw river
        cards_to_sample += 1;
        draw_only_river = true;
    }

    int won = 0;
    int lost = 0;
    int tied = 0;

    // -- START ONE ITERATION OF MC Simulation --
    // SAMPLE MISSING CARDS (opponent + board) FROM REMAINING DECK WITHOUT REPLACEMENT
    std::vector<int>::iterator sampled;
    std::sample(deck.begin(),deck.end(), sampled, cards_to_sample,
                std::mt19937{std::random_device{}()});

    // DRAW OPPONENT CARDS
    std::size_t idx_drawn = 0;
    int opponent_cards[n_opponents * 2];

    for(std::size_t opp = 0; opp<n_opponents; opp++) {
       opponent_cards[2*opp] = sampled[idx_drawn];
       idx_drawn++;
       opponent_cards[2*opp + 1] =  sampled[idx_drawn];
       idx_drawn++;
    }

    // DRAW BOARD CARDS
    if (draw_after_preflop) {
        cur_b0 = sampled[idx_drawn];
        idx_drawn ++;
        cur_b1 = sampled[idx_drawn];
        idx_drawn ++;
        cur_b2 = sampled[idx_drawn];
        idx_drawn ++;
        cur_b3 = sampled[idx_drawn];
        idx_drawn ++;
        cur_b4 = sampled[idx_drawn];
        idx_drawn ++;
    }
    else if (draw_turn_and_river) {
        cur_b0 = b0;
        cur_b1 = b1;
        cur_b2 = b2;
        cur_b3 = sampled[idx_drawn];
        idx_drawn ++;
        cur_b4 = sampled[idx_drawn];
        idx_drawn ++;
    }
    else if (draw_only_river) {
        cur_b0 = b0;
        cur_b1 = b1;
        cur_b2 = b2;
        cur_b3 = b3;
        cur_b4 = sampled[idx_drawn];
        idx_drawn ++;
    } else {
        cur_b0 = b0;
        cur_b1 = b1;
        cur_b2 = b2;
        cur_b3 = b3;
        cur_b4 = b4;
    }
    int hero_rank = SevenEval::GetRank(c0, c1, cur_b0, cur_b1, cur_b2,
                                       cur_b3, cur_b4);
    // rank[opp] = SevenEval::GetRank(c0, c1, b0, b1, b2, b3, b4);
    bool player_still_winning = true;
    int ties = 0;
    int opp_rank;
    for(std::size_t opp = 0; opp<n_opponents; opp++) {
        opp_rank = SevenEval::GetRank(opponent_cards[2*opp], opponent_cards[2*opp+1],
                                      cur_b0, cur_b1, cur_b2,cur_b3, cur_b4);
        if (opp_rank > hero_rank) {
            player_still_winning = false;
            break;
        } else if (opp_rank == hero_rank) {
            ties++;
        }
    }
    if (!player_still_winning) {
        lost += 1;
    }
    else if (player_still_winning && (ties < n_opponents)) {
        won += 1;
    }
    else if (player_still_winning && (ties == n_opponents)) {
        tied += 1;
    }
    else {
        throw std::logic_error("Hero can tie against at most n_opponents, not more. "
                               "Aborting MC Simulation...");
    }
    return SevenEval::GetRank(c0, c1, b0, b1, b2, b3, b4);
}

PYBIND11_MODULE(hand_eval, handle) {
    handle.doc() = "This function runs monte carlo simulation to determine a 7-card rank.";
    handle.def("run_monte_carlo", &run_mc);
}