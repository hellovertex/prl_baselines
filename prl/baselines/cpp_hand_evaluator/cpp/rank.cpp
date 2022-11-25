#include <iostream>
#include <random>
#include <string>
#include <iterator>
#include <algorithm>
#include "ext/src/SevenEval.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

int rank(int c0, int c1,  // player cards
           int b0, int b1, int b2, int b3, int b4) {
    // Get the rank of the seven-card spade flush, ace high.
    // return SevenEval::GetRank(1, 41, 18, 19, 16, 20, 24);
    return SevenEval::GetRank(c0, c1, b0, b1, b2, b3, b4);
}

PYBIND11_MODULE(hand_evaluator, handle) {
    handle.doc() = "This function a 7-card Texas Hold-EM Poker card rank.";
    handle.def("rank", &rank);
}