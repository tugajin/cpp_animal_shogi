#include "game.hpp"
#include "movelist.hpp"
#include "movelegal.hpp"
#include "util.hpp"
#include "search.hpp"
#include "selfplay.hpp"
#include "ubfm.hpp"
#include "attack.hpp"
#include "matesearch.hpp"

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

int main(int argc, char *argv[]){
    init_table();
    //mate::test_mate();
    //movelist::test_move_list();
    //game::test_common();
    //attack::test_attack();
    gen::test_gen3();
    //gen::test_gen3();
    //search::test_search();
    //game::test_nn();
    //selfplay::test_selfplay();
    //ubfm::test_ubfm();
    //selfplay::execute_selfplay();
    return 0;
}