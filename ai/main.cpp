#include "game.hpp"
#include "movelist.hpp"
#include "movelegal.hpp"
#include "util.hpp"
#include "search.hpp"
#include "selfplay.hpp"
#include "ubfm.hpp"
#include "attack.hpp"
#include "matesearch.hpp"
#include "hash.hpp"
#include "nn.hpp"
#include "countreward.hpp"

TeeStream Tee;

int g_color_piece_no[COLOR_PIECE_END];
int g_piece_no_color_piece[12];
int g_piece_color_piece[COLOR_SIZE][12];

Square g_delta_inc_all[DELTA_NB];
ColorPiece g_delta_mask[DELTA_NB];

namespace ubfm {
UBFMSearcherGlobal g_searcher_global;
}
namespace selfplay {
ReplayBuffer g_replay_buffer;
}

int main(int /*argc*/, char **/*argv*/){
    check_mode();
    init_table();
    search::test_search();
    //selfplay::execute_selfplay();
    return 0;
}