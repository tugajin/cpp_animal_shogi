#ifndef __SELFPLAY_HPP__
#define __SELFPLAY_HPP__

#include <vector>
#include <utility>
#include <thread>
#include <fstream>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include "nlohmann/json.hpp"
#include "util.hpp"
#include "common.hpp"
#include "game.hpp"
#include "ubfm.hpp"
#include "nn.hpp"
#include "matesearch.hpp"
#include "countreward.hpp"

namespace selfplay {

using json = nlohmann::json;

class ReplayBuffer {
public:
    ReplayBuffer() {
        this->info = {};
    }
    void open() {
        this->info.clear();
        std::filesystem::create_directory("data");
        auto filename = "./data/" + timestamp() + "_" + to_string(my_rand(9999)) + ".json";
        this->ofs.open(filename);
    }
    void close() {
        this->ofs.close();
        this->info.clear();
    }
    void push_back(const Key hash, const nn::NNScore sc, const nn::NNScore delta) {
        info.push_back({{"p", hash},{"s", sc},{"d", delta}});
    }
    void write_data() {
        this->ofs<<this->info.dump();
    }
private:
    json info;
    std::ofstream ofs;
};

extern ReplayBuffer g_replay_buffer;

void push_back(const Key hash, const nn::NNScore score, const nn::NNScore delta) {
    g_replay_buffer.push_back(hash, score, delta);
}

Move execute_descent(game::Position &pos, reward::CountReward &cw) {
    assert(ubfm::g_searcher_global.root_node.n == 0);
    ubfm::g_searcher_global.root_node.pos = pos;
    ubfm::g_searcher_global.run();
    ubfm::g_searcher_global.join();
    //ubfm::g_searcher_global.choice_best_move_e_greedy();
    ubfm::g_searcher_global.choice_best_move_count(cw);
    ubfm::g_searcher_global.add_replay_buffer(&ubfm::g_searcher_global.root_node);
    return ubfm::g_searcher_global.root_node.best_move;
}

void execute_selfplay() {
    
    ubfm::g_searcher_global.GPU_NUM = 1;
    ubfm::g_searcher_global.THREAD_NUM = 1;
    ubfm::g_searcher_global.IS_DESCENT = true;
    ubfm::g_searcher_global.TEMPERATURE = 0.2;
    
    ubfm::g_searcher_global.init();

    reward::CountReward cw;
    
    REP(i, INT_MAX) {
        game::Position pos;
        pos = hash::hirate();
        g_replay_buffer.open();
        while(true) {
            Tee<<"自己対局："<<i<<std::endl;
            Tee<<pos<<std::endl;
            ubfm::g_searcher_global.clear_tree();
            if (pos.is_lose() || pos.is_draw(4)) {
                if (pos.is_draw(4)) {Tee<<"引き分け\n";}
                if (pos.is_lose()) {Tee<<"負け\n";}
                g_replay_buffer.write_data();
                g_replay_buffer.close();
                break;
            }
            ubfm::g_searcher_global.DESCENT_PO_NUM = gen::num_legal(pos) * 10;
            
            auto best_move = execute_descent(pos, cw);
            
            Tee<<ubfm::g_searcher_global.root_node.w<<std::endl;

            cw.update(pos.history());

            if (!attack::in_checked(pos)) {
                const auto mate_move = mate::mate_search(pos,5);
                if (mate_move != MOVE_NONE) {
                    Tee<<"found mate\n";
                    best_move = mate_move;
                }
            } 
            Tee<<move_str(best_move)<<std::endl;
            pos = pos.next(best_move);
        }
        if (true) {
            Tee<<"\n";
            ubfm::g_searcher_global.load_model();
            cw.dump();
        }
        Tee<<".";
    }
}
void test() {
}
void test_selfplay() {
    std::thread th_a(test);
    th_a.join();
}

}
#endif