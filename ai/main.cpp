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
#include "oracle.hpp"
#include "reviewbuffer.hpp"
#include "model.hpp"

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
SelfPlayWorker g_selfplay_worker[SelfPlayWorker::NUM];
int g_thread_counter;
SelfPlayInfo g_selfplay_info;
}
namespace oracle {
OracleData g_oracle;
}
namespace review {
ReviewBuffer g_review_buffer;
}
namespace model {
GPUModel g_gpu_model[GPUModel::GPU_NUM];
}
int main(int argc, char **argv){
    auto num = 999999999;
    if (argc > 1) {
        num = std::stoi(std::string(argv[1]));
    }
    check_mode();
    init_table();
    //oracle::g_oracle.load();
    model::g_gpu_model[0].load_model(0);
    //model::test_oracle_model();
    selfplay::execute_selfplay(num);
    //review::test_review();
    //model::test_model();
    return 0;
}