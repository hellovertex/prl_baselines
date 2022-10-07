#include <iostream>
#include "ext/src/SevenEval.h"

int main() {
    // Get the rank of the seven-card spade flush, ace high.
    std::cout << SevenEval::GetRank(1, 41, 18, 19, 16, 20, 24) << std::endl;
    return 0;
}