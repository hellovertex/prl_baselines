#include <iostream>
#include <random>
#include <string>
#include <iterator>
#include <algorithm>
#include "ext/src/SevenEval.h"
//#include <pybind11/pybind11.h>




//namespace py = pybind11;
//std::vector<int>
const int _deck_size_45 = 45;
const int _deck_size_46 = 46;
const int _deck_size_47 = 47;
const int _deck_size_48 = 48;
const int _deck_size_49 = 49;
const int _deck_size_50 = 50;

const int _cards_to_sample2 = 2;
const int _cards_to_sample3 = 3;
const int _cards_to_sample4 = 4;
const int _cards_to_sample5 = 5;
const int _cards_to_sample6 = 6;
const int _cards_to_sample7 = 7;
const int _cards_to_sample8 = 8;
const int _cards_to_sample9 = 9;
const int _cards_to_sample10 = 10;
const int _cards_to_sample11 = 11;
const int _cards_to_sample12 = 12;
const int _cards_to_sample13 = 13;
const int _cards_to_sample14 = 14;
const int _cards_to_sample15 = 15;

std::array<int8_t, _deck_size_45> deck45;  // 52 - hero hand and flop + turn + river cards
std::array<int8_t, _deck_size_46> deck46;  // 52 - hero hand and flop + turn card
std::array<int8_t, _deck_size_47> deck47;  // 52 - hero hand and flop
std::array<int8_t, _deck_size_50> deck50;  // 52 - hero hand

int run_mc(int c0, int c1,  // player cards
           int b0, int b1, int b2, int b3, int b4, // board cards or -1
           int n_opponents, // number of opponents
           int n_iters) {  // number of iterations to run mc
    // Get the rank of the seven-card spade flush, ace high.
    // std::cout << SevenEval::GetRank(1, 41, 18, 19, 16, 20, 24)  << std::endl;
    std::cout << "DEBUG\n";
    // BUILD REMAINING DECK - REMOVE DRAWN CARDS FROM DECK
    u_int8_t n_board_cards;
    int deck_size;
    int cards_to_sample = 2;

    bool draw_after_preflop = false;
    bool draw_turn_and_river = false;
    bool draw_only_river = false;

    // to store board of current simulation round in case of missing board cards
    int8_t cur_b0;
    int8_t cur_b1;
    int8_t cur_b2;
    int8_t cur_b3;
    int8_t cur_b4;

    int won = 0;
    int lost = 0;
    int tied = 0;

    // for fast runtime, we have 5 analogeous functions per number of sizes of remaining_deck
    if (b4!=-1) { // flop turn river card already given --> 45 cards remaining
        deck_size = 45;
        std::array deck = deck45;
        std::size_t idx = 0;
        for (std::size_t i=0; i<deck_size; i++) {
            if (i!=c0 && i!=c1 && i!=b0 &&i!=b1 && i!=b2 && i!=b3 && i!=b4){
                deck[idx] = i;
                idx++;
            }
        }
        cards_to_sample = n_opponents * 2 + 5;
        // -- START ONE ITERATION OF MC Simulation --
        for (int i = 0; i < n_iters; i++) {
            // SAMPLE MISSING CARDS (opponent + board) FROM REMAINING DECK WITHOUT REPLACEMENT
            std::array<int8_t, _cards_to_sample3>::iterator sampled;
            std::sample(deck.begin(),deck.end(), sampled, cards_to_sample,
                        std::mt19937{std::random_device{}()});
            // DRAW OPPONENT CARDS
            idx_drawn = 0;
            int8_t opponent_cards[n_opponents * 2];
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
            u_int16_t hero_rank = SevenEval::GetRank(c0, c1, cur_b0, cur_b1, cur_b2,
                                                     cur_b3, cur_b4);
            // count wins and loses
            bool player_still_winning = true;
            int ties = 0;
            u_int16_t opp_rank;
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
            std::cout << "Won" << won;
        }


    }
    else if (b3!=-1) {  // flop turn card already given --> 46 cards remaining
        deck_size = 46;
        std::array deck = deck46;
        std::size_t idx = 0;
        for (std::size_t i=0; i<deck_size; i++) {
            if (i!=c0 && i!=c1 && i!=b0 &&i!=b1 && i!=b2 && i!=b3 && i!=b4){
                deck[idx] = i;
                idx++;
            }
        }
        cards_to_sample = n_opponents * 2 + 4;
    }
    else if (b2!=-1) {  // flop cards already given --> 47 cards remaining
        deck_size = 47;
        std::array deck = deck47;
        std::size_t idx = 0;
        for (std::size_t i=0; i<deck_size; i++) {
            if (i!=c0 && i!=c1 && i!=b0 &&i!=b1 && i!=b2 && i!=b3 && i!=b4){
                deck[idx] = i;
                idx++;
            }
        }
        cards_to_sample = n_opponents * 2 + 3;
    }
    else {  // no board cards given, only hero cards --> 50 cards remain
        deck_size = 50;
        std::array deck = deck50;
        std::size_t idx = 0;
        for (std::size_t i=0; i<deck_size; i++) {
            if (i!=c0 && i!=c1 && i!=b0 &&i!=b1 && i!=b2 && i!=b3 && i!=b4){
                deck[idx] = i;
                idx++;
            }
        }
        cards_to_sample = n_opponents * 2;
    }


    std::size_t idx_drawn = 0;
    std::cout << "Won" << won << "\n";
    // std::vector<int> ret {won, lost, tied};
    // todo: before returning above expression,
    //  we need to find the cause of the signal 11: SIGSEGV
    return 0;
}

//PYBIND11_MODULE(hand_evaluator, handle) {
//    handle.doc() = "This function runs monte carlo simulation to determine a 7-card rank.";
//    handle.def("run_monte_carlo", &run_mc);
//}