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
#include "cns.hpp"
#include "bench.hpp"

TeeStream Tee;

int g_color_piece_no[COLOR_PIECE_END];
int g_piece_no_color_piece[12];
int g_piece_color_piece[COLOR_SIZE][12];

Square g_delta_inc_all[DELTA_NB];
ColorPiece g_delta_mask[DELTA_NB];

namespace ubfm {
UBFMSearcherGlobal g_searcher_global;
}
namespace cns {
CNSSearcherGlobal g_searcher_global;
}
namespace selfplay {
SelfPlayWorker g_selfplay_worker[SelfPlayWorker::NUM];
int g_thread_counter;
SelfPlayInfo g_selfplay_info;
int g_selfplay_po = 100;
std::vector<ProbeItem> g_probe_items;
int g_probe_po = 100;
std::vector<DumpItem> g_dump_items;
int g_dump_po = 10;
}
namespace oracle {
OracleData g_oracle;
}
namespace review {
ReviewBuffer g_review_buffer;
}
namespace model {
GPUModel g_gpu_model[GPUModel::GPU_NUM];
BatchPredictor g_batch_predictor;
}
int main(int argc, char **argv){
    auto num = 999999999;
    int selfplay_po = 100;
    std::string mode = "selfplay";
    std::string probe_keys;
    int probe_po = 100;
    std::string dump_keys;
    int dump_po = 10;
    std::string dump_out = "dump_trees.txt";
    if (argc > 1) {
        std::string arg1(argv[1]);
        if (arg1 == "bench") {
            mode = "bench";
        } else if (arg1 == "probe") {
            mode = "probe";
            if (argc > 2) {
                probe_keys = std::string(argv[2]);
            }
            if (argc > 3) {
                probe_po = std::stoi(argv[3]);
            }
        } else if (arg1 == "dump") {
            mode = "dump";
            if (argc > 2) {
                dump_keys = std::string(argv[2]);
            }
            if (argc > 3) {
                dump_po = std::stoi(argv[3]);
            }
            if (argc > 4) {
                dump_out = std::string(argv[4]);
            }
        } else {
            num = std::stoi(arg1);
            if (argc > 2) {
                selfplay_po = std::stoi(argv[2]);
            }
        }
    }
    check_mode();
    init_table();
    //oracle::g_oracle.load();
    model::g_gpu_model[0].load_model(0);

    if (mode == "bench") {
        bench::run_bench();
    } else if (mode == "probe") {
        if (probe_keys.empty()) {
            Tee<<"usage: cpp_animal_shogi probe <keys_file> [po_num]\n";
            return 1;
        }
        selfplay::execute_probe(probe_keys, probe_po);
    } else if (mode == "dump") {
        if (dump_keys.empty()) {
            Tee<<"usage: cpp_animal_shogi dump <keys_file> [po_num] [out_file]\n";
            return 1;
        }
        selfplay::execute_dump(dump_keys, dump_po, dump_out);
    } else {
        selfplay::execute_selfplay(num, selfplay_po);
    }
    return 0;
}