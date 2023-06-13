#ifndef __SELFPLAY_HPP__
#define __SELFPLAY_HPP__

#include <vector>
#include <utility>
#include <thread>
#include <fstream>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "util.hpp"
#include "common.hpp"
#include "game.hpp"
#include "ubfm.hpp"
#include "nn.hpp"
#include "matesearch.hpp"
#include "countreward.hpp"
#include "oracle.hpp"
#include "reviewbuffer.hpp"

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
    void push_back(const Key hash, const nn::NNScore sc) {
        info.push_back({{"p", hash},{"s", sc},{"r", -99}});
    }
    void overwrite_result(double result) {
        for(auto &item :this->info) {
            item["r"] = result;
            result *= -1;
        }
    }
    void write_data() {
        this->ofs<<this->info.dump();
    }
private:
    json info;
    std::ofstream ofs;
};
class DescentSearcherLocal ;

class SelfPlayWorker : public ubfm::UBFMSearcherGlobal {
public:
    static constexpr int NUM = 4;
    void run_descent(const int selfplay_num);
    void join();
    void init();
private:
    std::vector<DescentSearcherLocal> worker;
};

class DescentSearcherLocal : public ubfm::UBFMSearcherLocal {
public:
    DescentSearcherLocal(const int id, const int gpu_id, SelfPlayWorker * selfplay) :
    ubfm::UBFMSearcherLocal(id, gpu_id, selfplay),cw(reward::CountReward(id)) {
    }
    void run_descent(const int selfplay_num);
    void search_descent(const uint32 simulation_num);
    void selfplay(const int selplay_num);
    ReplayBuffer replay_buffer;
    reward::CountReward cw;
private:
    void evaluate_descent(ubfm::Node *node);
    Move execute_descent(game::Position &pos);
    bool interrupt_descent(const uint32 current_num, const uint32 simulation_num) const;
    void choice_best_move_count();
    int po_num = 100;
};

extern SelfPlayWorker g_selfplay_worker[SelfPlayWorker::NUM];
extern int g_thread_counter;

void execute_selfplay(const int num) {
    g_thread_counter = 0;
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].init();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].run_descent(num);
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].join();
    }
}

void SelfPlayWorker::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    this->worker.emplace_back(g_thread_counter,0,this);
    this->clear_tree();
    g_thread_counter++;
}

void SelfPlayWorker::run_descent(const int selfplay_num) {
    ASSERT(this->worker.size() > 0);
    this->worker[0].run_descent(selfplay_num);
}

void SelfPlayWorker::join() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].join();
}

void DescentSearcherLocal::run_descent(const int selfplay_num) {
    this->thread = new std::thread([this, selfplay_num]() {
        this->selfplay(selfplay_num);
    });
}
void DescentSearcherLocal::search_descent(const uint32 simulation_num) {
    for(auto i = 0u ;; ++i) {
        //Tee<<"start simulation:" << i <<"/"<<simulation_num<<"\r";
        const auto interrupt = this->interrupt_descent(i, simulation_num);
        if (interrupt) {
            break;
        }
        this->evaluate_descent(this->root_node());
    }
}

void DescentSearcherLocal::evaluate_descent(ubfm::Node *node) {
    //ASSERT(!node->is_resolved());
    node->n++;
    // Timer timer;
    // timer.start();
    ASSERT2(node->pos.is_ok(),{
        Tee<<node->pos<<std::endl;
    })
    ASSERT(std::fabs(node->w) <= 1);
    ASSERT2(node->is_ok(),{
        Tee<<node->str()<<std::endl;
    });
    
    if (node->pos.is_draw()) {
        node->w = nn::NNScore(0.0);
        node->state = ubfm::NodeState::NodeDraw;
        return;
    } 
    // if (node->pos.ply() == 0)
    //     Tee << "draw:" << timer.elapsed()<<std::endl;
    if (node->pos.is_lose()) {
        node->w = ubfm::score_lose(node->ply);
        node->state = ubfm::NodeState::NodeLose;
        return;
    }
    // if (node->pos.ply() == 0)
    //     Tee << "lose:" << timer.elapsed()<<std::endl;
    if (node->is_terminal()) {
        this->expand(node);
        // if (node->pos.ply() == 0)
        //     Tee << "expand:" << timer.elapsed()<<std::endl;
        this->predict(node);
        // if (node->pos.ply() == 0)
        //     Tee << "predict:" << timer.elapsed()<<std::endl;
    } else {
        // if (node->pos.ply() == 0)
        //     Tee << "is_resolved:" << timer.elapsed()<<std::endl;
        auto next_node = this->next_child<true>(node);
        // if (node->pos.ply() == 0)
        //     Tee << "next_child:" << timer.elapsed()<<std::endl;
        ASSERT2(next_node != nullptr,{
            Tee<<node->str()<<std::endl;
        })
        this->evaluate_descent(next_node);
        // if (node->pos.ply() == 0)
        //     Tee << "evaluate_descent:" << timer.elapsed()<<std::endl;
        // if (node->pos.ply() == 0)
        //     Tee << "update_node:" << timer.elapsed()<<std::endl;
    }
    this->update_node(node);
}

bool DescentSearcherLocal::interrupt_descent(const uint32 current_num, const uint32 simulation_num) const {
    if (this->root_node()->is_resolved()) {
        return true;
    }
    // 最大基準の回数の2倍まではやってみる
    if (current_num >= simulation_num) {
        //Tee<<"over\n";
        return true;
    }
    auto mated_num = 0;
    auto mate_num = 0;
    auto root_node = this->root_node();
    REP(i, root_node->child_len) {
        const auto child = root_node->child(i);
        if (-child->w <= -0.9) {
            mated_num++;
        }
        if (-child->w >= 0.9) {
            mate_num++;
        }
    }
    // 最低2回くらいはやる
    const auto can_stop = current_num > static_cast<uint32>(2 * root_node->child_len);
    // 全部負けなら終わり
    if (can_stop && mated_num == root_node->child_len) {
        return true;
    }
    // 1手以外は全部負けなら終わり
    if (can_stop && mated_num + 1 == root_node->child_len) {
        return true;
    }
    //勝ちを見つけたら終わり
    if (can_stop && mate_num > 0) {
        return true;
    }
    if (current_num >= simulation_num) {
        nn::NNScore max_score = static_cast<nn::NNScore>(-1.0);
        auto max_num = -1;
        auto max_score_index = -1;
        auto max_num_index = -1;
        REP(i, root_node->child_len) {
            ASSERT(i>=0);
            ASSERT(i<root_node->child_len);
            const auto child = root_node->child(i);
            if (-child->w > max_score) {
                max_score = -child->w;
                max_score_index = i;
            }
            if (child->n > max_num) {
                max_num = child->n;
                max_num_index = i;
            }
        }
        // 一番選択した手と一番勝ちが高い手が不一致なら追加探索
        if (max_score_index != max_num_index) {
            //Tee<<"+ ";
            return false;
        }
        return true;
    }
    return false;
}

Move DescentSearcherLocal::execute_descent(game::Position &pos) {
    this->root_node()->pos = pos;
    this->search_descent(this->po_num);
    this->choice_best_move_count();
    const auto k = hash::hash_key(this->root_node()->pos);
    const auto w = this->root_node()->w;
    this->replay_buffer.push_back(k,w);
   return this->root_node()->best_move;
}

void DescentSearcherLocal::choice_best_move_count() {
    std::vector<double> scores;
    std::vector<uint64> num;
    REP(i, this->root_node()->child_len) {

        ASSERT(i>=0);
        ASSERT(i<this->root_node()->child_len);
        auto child = this->root_node()->child(i);
        const auto r = 1 + cw.get(child->pos.history());
        scores.push_back((1 / std::sqrt(r)));
        num.push_back(r);
    }
    REP(i, this->root_node()->child_len) {
        ASSERT(i>=0);
        ASSERT(i<this->root_node()->child_len);
        auto child = this->root_node()->child(i);
        const auto reward = scores[i] * 0.8;
        //const auto oracle = -oracle::g_oracle.result(child->pos);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = 0;
                }
                scores[i] = 1;
                break;
            } else if (child->is_draw()) {
                scores[i] = 0;
            } else if (child->is_win()) {
                scores[i] = -1.0 ;
            }
        } else {
            scores[i] = /*child->n + 1.0 */- child->w + reward;
        }
        // Tee<<"n:"<<padding_str(to_string(child->n),3) 
        //     << " w:" << padding_str(to_string(-child->w),7) 
        //     << " oracle:" << padding_str(to_string(oracle),2) 
        //     << " org:" << padding_str(to_string(num[i]),3)<<" "
        //     <<move_str(child->parent_move)<<std::endl;
    }
    auto index = -1;
    auto iter = std::max_element(scores.begin(), scores.end());
    index = std::distance(scores.begin(), iter);

    ASSERT(index>=0);
    ASSERT(index<this->root_node()->child_len);
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
    this->root_node()->w = -child->w;
}

void DescentSearcherLocal::selfplay(const int selplay_num) {
    REP(i, selplay_num) {
        game::Position pos;
        auto base_po = 10u;
        pos = hash::hirate();
        this->replay_buffer.open();
        while(true) {
            //Tee<<"自己対局("<<this->thread_id<<")："<<i<<":"<<pos.ply()<<std::endl;
            //Tee<<pos<<std::endl;
            
            this->global->clear_tree();

            if (pos.is_lose() || pos.is_draw(4)) {
                auto result = 0.0;
                if (pos.is_draw(4)) {
                    result = 0.0;
                    Tee<<"=("<<this->thread_id<<")";
                }
                if (pos.is_lose()) {
                    if (pos.turn() == BLACK) {
                        result = -1.0;
                        Tee<<"x("<<this->thread_id<<")";
                    } else {
                        result = 1.0;
                        Tee<<"o("<<this->thread_id<<")";
                    }
                }
                this->replay_buffer.overwrite_result(result);
                this->replay_buffer.write_data();
                this->replay_buffer.close();
                break;
            }
            this->po_num = 50;
            
            auto best_move = execute_descent(pos);
            
            this->cw.update(pos.history());
            if (!attack::in_checked(pos)) {
                const auto mate_move = mate::mate_search(pos,5);
                if (mate_move != MOVE_NONE) {
                    best_move = mate_move;
                }
            } 
            pos = pos.next(best_move);
        }
        if (i % 10 == 0) {
            //Tee<<"\n";
            cw.dump();
        }
    }
}
}
#endif