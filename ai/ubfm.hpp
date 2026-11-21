#ifndef __UBFM_HPP__
#define __UBFM_HPP__

#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include "common.hpp"
#include "util.hpp"
#include "game.hpp"
#include "thread.hpp"
#include "nn.hpp"
#include "countreward.hpp"
#include "oracle.hpp"
#include "model.hpp"

namespace ubfm {

inline bool g_use_batch_predict = false;
inline bool g_use_cn_search = true;  // true で葉選択を共謀数（CN）基準にする（生成実験用）

enum NodeState : int {
    NodeDraw = 0,
    NodeWin = 1,
    NodeLose = 2,
    NodeUnknown = -1
};

inline nn::NNScore score_win(const int ply) {
    return nn::NNScore(0.9999) - nn::NNScore(static_cast<double>(ply) / 2000.0);
}
inline nn::NNScore score_lose(const int ply) {
    return -score_win(ply);
}

class Node {
public:
    Node() : child_nodes(nullptr),
             parent_move(MOVE_NONE),
             best_move(MOVE_NONE),
             state(NodeUnknown),
             w(0.0),
             init_w(0.0),
             n(0),
             vl(0),
             child_len(-1),
             ply(-1),
             cn_cost{1, 0, 0, 1}{}
   Node( const Node & ) = delete ;
   Node & operator = ( const Node & ) = delete ;
   Node & operator = ( const Node && n) = delete;
    void init() {
        this->w = this->init_w = 0.0;
        this->n = 0;
        this->vl = 0;
        this->parent_move = this->best_move = MOVE_NONE;
        this->child_len = -1;
        this->child_nodes = nullptr;
        this->ply = 0;
        this->state = NodeUnknown;
        this->cn_cost[0] = 1;
        this->cn_cost[1] = 0;
        this->cn_cost[2] = 0;
        this->cn_cost[3] = 1;
    }
    bool is_resolved() const {
        return state != NodeUnknown;
    }
    bool is_win() const {
        return state == NodeWin;
    }
    bool is_lose() const {
        return state == NodeLose;
    }
    bool is_draw() const {
        return state == NodeDraw;
    }
    bool is_terminal() const {
        return child_len == -1;
    }

    std::string str(const bool is_root = true) const {
        const std::string padding = is_root ? "" :"        ";
        std::string str = "---------------------------\n";
        if (is_root) { str += pos.str(); }
        str += padding + "w:" + to_string(w) + "\n";
        str += padding + "init_w:" + to_string(init_w) + "\n";
        str += padding + "n:" + to_string(n) + "\n";
        str += padding + "child_len:" + to_string(child_len) + "\n";
        str += padding + "ply:" + to_string(ply) + "\n";
        str += padding + "parent_move:" + move_str(parent_move) + "\n";
        str += padding + "best_move:" + move_str(best_move) + "\n";
        str += padding + "state:" + to_string(state) + "\n";
        str += "---------------------------\n";
        if (is_root && !this->is_terminal()) {
            str += "child\n";
            REP(i, this->child_len) {
                str += "no:" + to_string(i) + "\n";
                ASSERT(i>=0);
                ASSERT(i<this->child_len);
                auto child = this->child(i);
                str += child->str(false);
            }
        }
        return str;
    }
    Node* child(const int index) const {
        ASSERT(index < child_len);
        ASSERT(index >= 0);
        return child_nodes[index].get();
    }
    bool is_ok() const {
        if (this->is_terminal()) {
            if (this->n != 1 && this->n != 0) {
                Tee<<"terminal error\n";
                return false;
            }
            return true;
        } else {
            const auto parent_n = this->n;
            auto child_n = 0;
            REP(i, this->child_len) {
                const auto child = this->child(i);
                child_n += child->n;
            }
            if (parent_n != (child_n+1)) {
                Tee<<"parent error\n";
                return false;
            }
            return true;
        }
    }
    bool is_ok2() const {
        if (this->is_terminal()) {
            // こないはず
            Tee<<"terminal error2\n";
            return false;
        }
        const auto parent_n = this->n;
        auto child_n = 0;
        REP(i, this->child_len) {
            const auto child = this->child(i);
            child_n += child->n;
        }
        if (parent_n != child_n) {
            Tee<<"parent error2\n";
            return false;
        }
        return true;
    }
    game::Position pos;
    std::unique_ptr<std::unique_ptr<Node>[]> child_nodes;
    Lockable lock_node;
    Move parent_move;
    Move best_move;
    NodeState state;
    nn::NNScore w;
    nn::NNScore init_w;
    int n;
    int vl;
    int child_len;
    int ply;
    // 共謀数（compute_cn が書く）: [0]=GE(+0.2) [1]=LE(+0.2) [2]=GE(-0.2) [3]=LE(-0.2)
    // 初期値は w=0 の葉としての値（GE(+τ)=1, LE(+τ)=0, GE(-τ)=0, LE(-τ)=1）
    int cn_cost[4];
};

// ---- 共謀数（カテゴリ境界 ±0.2 に対する threshold conspiracy number） ----
// GE(τ): そのノードの値を τ より大きくするために必要な「葉評価の変更」の最小数
// LE(τ): そのノードの値を τ より小さくするために必要な「葉評価の変更」の最小数
// 葉は未展開・未解決ノードで変更コスト 1。解決済みノードは葉の変更では動かない（CN_INF）。
// 内部ノードは w = max_i(-子.w) なので、
//   GE(τ) = min_i LE_i(-τ) / LE(τ) = Σ_i GE_i(-τ) と再帰する。
static constexpr double CN_TAU = 0.2;
static constexpr int CN_INF = 1 << 28;

inline int cn_add_cap(const int a, const int b) {
    const long v = static_cast<long>(a) + static_cast<long>(b);
    return v >= CN_INF ? CN_INF : static_cast<int>(v);
}

inline void compute_cn(Node *node) {
    const auto w = node->w;
    if (node->is_resolved()) {
        node->cn_cost[0] = (w >  CN_TAU) ? 0 : CN_INF;
        node->cn_cost[1] = (w <  CN_TAU) ? 0 : CN_INF;
        node->cn_cost[2] = (w > -CN_TAU) ? 0 : CN_INF;
        node->cn_cost[3] = (w < -CN_TAU) ? 0 : CN_INF;
        return;
    }
    if (node->is_terminal()) {
        node->cn_cost[0] = (w >  CN_TAU) ? 0 : 1;
        node->cn_cost[1] = (w <  CN_TAU) ? 0 : 1;
        node->cn_cost[2] = (w > -CN_TAU) ? 0 : 1;
        node->cn_cost[3] = (w < -CN_TAU) ? 0 : 1;
        return;
    }
    auto ge_p = CN_INF;  // GE(+τ) = min_i LE_i(-τ)
    auto ge_m = CN_INF;  // GE(-τ) = min_i LE_i(+τ)
    auto le_p = 0;       // LE(+τ) = Σ_i GE_i(-τ)
    auto le_m = 0;       // LE(-τ) = Σ_i GE_i(+τ)
    REP(i, node->child_len) {
        auto child = node->child(i);
        compute_cn(child);
        ge_p = std::min(ge_p, child->cn_cost[3]);
        ge_m = std::min(ge_m, child->cn_cost[1]);
        le_p = cn_add_cap(le_p, child->cn_cost[2]);
        le_m = cn_add_cap(le_m, child->cn_cost[0]);
    }
    node->cn_cost[0] = ge_p;
    node->cn_cost[1] = le_p;
    node->cn_cost[2] = ge_m;
    node->cn_cost[3] = le_m;
}

// 根のカテゴリ（win/draw/lose）を変えるのに必要な葉評価の変更数。
// 前提: compute_cn(root) 済み。CN_INF は「葉をどう変えてもカテゴリは変わらない（証明済み級）」。
inline int root_category_cn(const Node *root) {
    if (root->w > CN_TAU) {
        return root->cn_cost[1];
    }
    if (root->w < -CN_TAU) {
        return root->cn_cost[2];
    }
    return std::min(root->cn_cost[0], root->cn_cost[3]);
}

struct VLLeafInfo {
    Node *leaf;
    std::vector<Node*> path;
    bool needs_predict;
};

class UBFMSearcherGlobal;

class UBFMSearcherLocal {
public:
    UBFMSearcherLocal(int id, int gpu_id, UBFMSearcherGlobal * global) :
                     global(global),
                     thread(nullptr),
                     thread_id(id),
                     gpu_id(gpu_id) {
    }
    void search(const uint32 simulation_num);
    void search_descent(const uint32 simulation_num);
    void selfplay() {};
    bool is_ok();
    void run();
    void join();
protected:
    void evaluate(Node *node);
    void predict(Node *node);
    void expand(Node *node);
    Node *next_child(const Node *node) const;
    Node *next_child_vl(const Node *node) const;
    VLLeafInfo select_leaf_cn(Node *root);
    void update_node(Node *node);
    bool interrupt(const uint32 current_num, const uint32 simulation_num) const;
    Node *root_node() const ;
    VLLeafInfo select_leaf_vl(Node *root);
    void predict_batch_leaves(std::vector<VLLeafInfo> &leaves);
    void backup_vl(std::vector<VLLeafInfo> &leaves);
    static constexpr int VL_BATCH_SIZE = 8;
    static constexpr double VL_WEIGHT = 0.5;

    UBFMSearcherGlobal *global;
    std::thread *thread;
    int thread_id;
    int gpu_id;
};

class UBFMSearcherGlobal {
public:
    UBFMSearcherGlobal() :
                       THREAD_NUM(1){}
    UBFMSearcherGlobal(const int thread_num) : 
                       THREAD_NUM(thread_num){}
    Node root_node;
    void init();
    void clear_tree();
    void run();
    void join();
    void choice_best_move();

    int THREAD_NUM;
    int SIMULATION_NUM = 2000;
protected:
    std::vector<UBFMSearcherLocal> worker;
};

extern UBFMSearcherGlobal g_searcher_global;

Move think_ubfm(game::Position &pos) {
    ASSERT(g_searcher_global.root_node.n == 0);
    g_searcher_global.root_node.pos = pos;
    g_searcher_global.run();
    g_searcher_global.join();
    g_searcher_global.choice_best_move();
    return g_searcher_global.root_node.best_move;
}

Node *UBFMSearcherLocal::root_node() const {
    return &this->global->root_node;
}

void UBFMSearcherGlobal::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    REP(i, UBFMSearcherGlobal::THREAD_NUM) {
        this->worker.emplace_back(i,0,this);
    }
    this->clear_tree();
}

void UBFMSearcherGlobal::clear_tree() {
    this->root_node.init();
}

void UBFMSearcherGlobal::run() {
    REP(i, UBFMSearcherGlobal::THREAD_NUM) {
        this->worker[i].run();
    }
}

void UBFMSearcherGlobal::join() {
    REP(i, UBFMSearcherGlobal::THREAD_NUM) {
        this->worker[i].join();
    }
}

void UBFMSearcherGlobal::choice_best_move() {
    std::vector<double> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(0.0);
    }
    REP(i, this->root_node.child_len) {
        ASSERT(i>=0);
        ASSERT(i<this->root_node.child_len);
        auto child = this->root_node.child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node.child_len) {
                    scores[i] = 0;
                }
                scores[i] = 1;
                break;
            } else if (child->is_draw()) {
                scores[i] = (child->n) + 1.0;
            } else if (child->is_win()) {
                scores[i] = 0.0;
            }
        } else {
            scores[i] = child->n + 1.0 - child->w;
        }
    }
    auto index = -1;
    auto iter = std::max_element(scores.begin(), scores.end());
    index = std::distance(scores.begin(), iter);
    ASSERT(index>=0);
    ASSERT(index<this->root_node.child_len);
    auto child = this->root_node.child(index);
    this->root_node.best_move = child->parent_move;
}

void UBFMSearcherLocal::run() {
	this->thread = new std::thread([this]() {
        this->search(int(this->global->SIMULATION_NUM / this->global->THREAD_NUM));
    });
}
void UBFMSearcherLocal::join() {
    this->thread->join();
    delete this->thread;
}

bool UBFMSearcherLocal::interrupt(const uint32 current_num, const uint32 simulation_num) const {
    if (this->root_node()->is_resolved()) {
        return true;
    }
    if (current_num >= simulation_num) {
        return true;
    }
    return false;
}

void UBFMSearcherLocal::search(const uint32 simulation_num) {

    const auto is_out = (this->thread_id == 0) && (this->gpu_id == 0);
    for (auto i = 0u;;) {
        if (this->interrupt(i, simulation_num)) break;

        std::vector<VLLeafInfo> leaves;
        for (int b = 0; b < VL_BATCH_SIZE; b++) {
            if (this->root_node()->is_resolved()) break;
            leaves.push_back(this->select_leaf_vl(this->root_node()));
        }
        if (leaves.empty()) break;

        this->predict_batch_leaves(leaves);
        this->backup_vl(leaves);
        i += static_cast<uint32>(leaves.size());
    }
    if (is_out) {
        Tee<<this->root_node()->str()<<std::endl;
    }
}

void UBFMSearcherLocal::evaluate(Node *node) {
    ASSERT(node != nullptr);
    ASSERT(std::fabs(node->w) <= 1);
    node->lock_node.lock();
    node->n++;

    if (node->pos.is_draw()) {
        node->w = nn::NNScore(0.0);
        node->state = NodeState::NodeDraw;
        node->lock_node.unlock();
        return;
    } 
    if (node->pos.is_lose()) {
        node->state = NodeState::NodeLose;
        node->lock_node.unlock();
        return;
    }
    if (node->is_resolved()) {
        node->lock_node.unlock();
        return;
    }
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
        node->lock_node.unlock();
    } else {
        auto next_node = this->next_child(node);
        node->lock_node.unlock();
        this->evaluate(next_node);
    }
    node->lock_node.lock();
    this->update_node(node);

    node->lock_node.unlock();
}

void UBFMSearcherLocal::expand(Node *node) {
    auto moveList = movelist::MoveList();
    gen::legal_moves(node->pos, moveList);
    
    node->child_len = moveList.len();
    node->child_nodes = std::make_unique<std::unique_ptr<Node>[]>(node->child_len);
    REP(i, node->child_len) {
        auto next_pos = node->pos.next(moveList[i]);
        node->child_nodes[i] = std::make_unique<Node>();
        auto next_node = node->child_nodes[i].get();
        next_node->pos = next_pos;
        next_node->ply = node->ply+1;
        next_node->parent_move = moveList[i];
    }
}
void UBFMSearcherLocal::predict(Node *node) {
    
    ASSERT2(node->child_len > 0,{
        Tee<<node->pos<<std::endl;
    });
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> outputs_list;
    REP(i, node->child_len) {
        ASSERT(i>=0);
        ASSERT(i<node->child_len);
        auto child = node->child(i);
        auto &pos = child->pos;
        feat_list.push_back(nn::feature(pos));
    }
    if (g_use_batch_predict) {
        model::predict_batched(this->gpu_id, feat_list, outputs_list);
    } else {
        model::predict(this->gpu_id, feat_list, outputs_list);
    }
    
    REP(i, node->child_len) {
        auto state = NodeUnknown;
        auto score = outputs_list[i];
        if (score >= nn::NNScore(1.0)) {
            score = nn::NNScore(0.8999);
        } else if (score <= nn::NNScore(-1.0)) {
            score = nn::NNScore(-0.8999);
        }
        ASSERT(i>=0);
        ASSERT(i<node->child_len);

        auto child = node->child(i);
        auto &pos = child->pos;

        if (pos.is_draw()) {
            score = nn::NNScore(0.0);
            state = NodeDraw;
        } else if (pos.is_lose()) {
            score = score_lose(child->ply);
            state = NodeLose;
        } else if (pos.is_win()) {
            score = score_win(child->ply);
            state = NodeWin;
        }
#if 1
        else if (!attack::in_checked(pos)) {
            const auto move = mate::mate_search(pos,5);
            if (move != MOVE_NONE) {
                score = score_win(child->ply);
                state = NodeWin;
            }
        } else {
            const auto move = mate::mated_search(pos,4);
            if (move == MOVE_NONE) {
                score = score_lose(child->ply);
                state = NodeLose;
            }
        }
#endif
        ASSERT2(std::fabs(score)<=1,{
            Tee<<child->ply<<std::endl;
        });
        child->w = child->init_w = score;
        child->state = state;
    }
    //Tee<<"  end:"<<timer.elapsed()<<std::endl;
}
Node *UBFMSearcherLocal::next_child(const Node *node) const {
    ASSERT(node->child_len >= 0);
    Node *best_child = nullptr;
    nn::NNScore best_score = nn::NNScore(-10000);
    auto min_num = INT32_MAX;
    REP(i, node->child_len) {

        ASSERT(i>=0);
        ASSERT(i<node->child_len);

        auto child = node->child(i);
        ASSERT(std::fabs(child->w)<=1); 
        if (child->is_resolved()) { continue; }
        const auto score = -child->w + static_cast<nn::NNScore>(rand_gaussian(0.0,0.2));
        if (score > best_score) {
            best_score = score;
            min_num = child->n;
            best_child = child;
        } else if (score == best_score) {
            if (child->n < min_num ) {
                min_num = child->n;
                best_child = child;
            }
        }
    }
    return best_child;
}

void UBFMSearcherLocal::update_node(Node *node) {

    Node *best_child = nullptr;
    auto max_value = nn::NNScore(-1);
    auto max_num = -1;
    auto lose_num = 0;
    auto draw_num = 0;
    const auto child_len = node->child_len;
    ASSERT2(child_len > 0,{
        Tee<<node->pos<<std::endl;
        Tee<<child_len<<std::endl;
        Tee<<node->pos.is_done()<<std::endl;
        Tee<<node->str()<<std::endl;
    });

    REP(i, child_len) {
        ASSERT(i>=0);
        ASSERT(i<child_len);

        auto child = node->child(i);
        ASSERT(std::fabs(child->w)<=1);
        if (child->is_resolved()) {

            //子供に負けを見つけた→つまり勝ちなので終わり
            if (child->is_lose()) {

                best_child = child;
                node->state = NodeWin;
                node->w = -child->w;
                node->best_move = child->parent_move;
                
                return;
            } else if (child->is_win()) {
                lose_num++;
            } else {
                ASSERT(child->is_draw());
                draw_num++;
            }
        }

        if (-child->w > max_value) {
            best_child = child;
            max_value = -child->w;
            max_num = child->n;
        } else if (-child->w == max_value) {
            if (child->n > max_num) {
                best_child = child;
                max_value = -child->w;
                max_num = child->n;
            }
        }
    }
    ASSERT2(best_child != nullptr,{
        Tee<<node->pos<<std::endl;
        Tee<<child_len<<std::endl;
        Tee<<node->pos.is_done()<<std::endl;
        Tee<<node->str()<<std::endl;
    });
    
    if (child_len == draw_num) {
        node->state = NodeDraw;
        node->w = nn::NNScore(0.0);
        return;
    } else if (child_len == lose_num) {
        node->state = NodeLose;
        node->w = -best_child->w;
        node->best_move = best_child->parent_move;
        return;
    } else if (child_len == (draw_num + lose_num)) {
        node->state = NodeDraw;
        node->w = nn::NNScore(0.0);
        return;
    }
    node->w = -best_child->w;
    node->best_move = best_child->parent_move;
}

Node *UBFMSearcherLocal::next_child_vl(const Node *node) const {
    ASSERT(node->child_len >= 0);
    Node *best_child = nullptr;
    nn::NNScore best_score = nn::NNScore(-10000);
    auto min_num = INT32_MAX;
    REP(i, node->child_len) {
        auto child = node->child(i);
        ASSERT(std::fabs(child->w) <= 1);
        if (child->is_resolved()) { continue; }
        const auto score = -child->w
                         - nn::NNScore(child->vl * VL_WEIGHT)
                         + static_cast<nn::NNScore>(rand_gaussian(0.0, 0.2));
        if (score > best_score) {
            best_score = score;
            min_num = child->n;
            best_child = child;
        } else if (score == best_score) {
            if (child->n < min_num) {
                min_num = child->n;
                best_child = child;
            }
        }
    }
    return best_child;
}

VLLeafInfo UBFMSearcherLocal::select_leaf_vl(Node *root) {
    VLLeafInfo info;
    info.needs_predict = false;
    auto *node = root;

    while (true) {
        node->lock_node.lock();
        info.path.push_back(node);
        node->vl++;

        if (node->pos.is_draw()) {
            node->w = nn::NNScore(0.0);
            node->state = NodeState::NodeDraw;
            node->lock_node.unlock();
            info.leaf = node;
            return info;
        }
        if (node->pos.is_lose()) {
            node->w = score_lose(node->ply);
            node->state = NodeState::NodeLose;
            node->lock_node.unlock();
            info.leaf = node;
            return info;
        }
        if (node->is_resolved()) {
            node->lock_node.unlock();
            info.leaf = node;
            return info;
        }
        if (node->is_terminal()) {
            this->expand(node);
            info.needs_predict = true;
            info.leaf = node;
            node->lock_node.unlock();
            return info;
        }
        auto *next = this->next_child_vl(node);
        node->lock_node.unlock();
        if (next == nullptr) {
            info.leaf = node;
            return info;
        }
        node = next;
    }
}

// 共謀数による葉選択: 根のカテゴリを変える最小共謀（葉の集合）に属する葉を1つ選ぶ。
// 前提: compute_cn(root) 済み（cn_cost はバッチ開始時点の値。バッチ内の展開は次バッチで反映される）。
// 共謀を辿れない場合は info.leaf == nullptr を返し、呼び出し側が select_leaf_vl にフォールバックする。
VLLeafInfo UBFMSearcherLocal::select_leaf_cn(Node *root) {
    VLLeafInfo info;
    info.needs_predict = false;
    info.leaf = nullptr;
    // 根の目標方向: kind 0=GE / 1=LE、sign +1 は τ=+0.2、-1 は τ=-0.2
    int kind, sign;
    if (root->w > CN_TAU) {
        kind = 1; sign = +1;          // 勝ちカテゴリから落とす共謀
    } else if (root->w < -CN_TAU) {
        kind = 0; sign = -1;          // 負けカテゴリから上げる共謀
    } else if (root->cn_cost[0] <= root->cn_cost[3]) {
        kind = 0; sign = +1;          // draw から勝ちへ抜ける方が安い
    } else {
        kind = 1; sign = -1;          // draw から負けへ落ちる方が安い
    }
    auto *node = root;
    while (true) {
        node->lock_node.lock();
        info.path.push_back(node);
        node->vl++;

        if (node->pos.is_draw()) {
            node->w = nn::NNScore(0.0);
            node->state = NodeState::NodeDraw;
            node->lock_node.unlock();
            info.leaf = node;
            return info;
        }
        if (node->pos.is_lose()) {
            node->w = score_lose(node->ply);
            node->state = NodeState::NodeLose;
            node->lock_node.unlock();
            info.leaf = node;
            return info;
        }
        if (node->is_resolved()) {
            node->lock_node.unlock();
            info.leaf = node;
            return info;
        }
        if (node->is_terminal()) {
            this->expand(node);
            info.needs_predict = true;
            info.leaf = node;
            node->lock_node.unlock();
            return info;
        }
        Node *next = nullptr;
        auto best = CN_INF;
        if (kind == 0) {
            // GE(sign·τ): LE_child(-sign·τ) が最小の子へ（子の目標は LE(-sign·τ)）
            const auto idx = (sign > 0) ? 3 : 1;
            REP(i, node->child_len) {
                auto child = node->child(i);
                if (child->cn_cost[idx] < best) {
                    best = child->cn_cost[idx];
                    next = child;
                }
            }
        } else {
            // LE(sign·τ): GE_child(-sign·τ) が正で最小の子へ（子の目標は GE(-sign·τ)）
            const auto idx = (sign > 0) ? 2 : 0;
            REP(i, node->child_len) {
                auto child = node->child(i);
                const auto c = child->cn_cost[idx];
                if (c > 0 && c < best) {
                    best = c;
                    next = child;
                }
            }
        }
        node->lock_node.unlock();
        if (next == nullptr || best >= CN_INF) {
            // 共謀を辿れない: vl を巻き戻して失敗を返す
            for (auto *p : info.path) {
                p->lock_node.lock();
                p->vl--;
                p->lock_node.unlock();
            }
            info.path.clear();
            info.leaf = nullptr;
            return info;
        }
        kind = 1 - kind;
        sign = -sign;
        node = next;
    }
}

void UBFMSearcherLocal::predict_batch_leaves(std::vector<VLLeafInfo> &leaves) {
    std::vector<nn::Feature> all_features;
    std::vector<std::pair<size_t, int>> ranges;

    for (auto &info : leaves) {
        if (!info.needs_predict) continue;
        auto *node = info.leaf;
        ASSERT(node->child_len > 0);
        size_t start = all_features.size();
        REP(i, node->child_len) {
            auto child = node->child(i);
            all_features.push_back(nn::feature(child->pos));
        }
        ranges.push_back({start, node->child_len});
    }

    if (all_features.empty()) return;

    std::vector<nn::NNScore> all_outputs;
    if (g_use_batch_predict) {
        model::predict_batched(this->gpu_id, all_features, all_outputs);
    } else {
        model::predict(this->gpu_id, all_features, all_outputs);
    }

    size_t range_idx = 0;
    for (auto &info : leaves) {
        if (!info.needs_predict) continue;
        auto *node = info.leaf;
        auto [start, child_count] = ranges[range_idx++];

        REP(i, child_count) {
            auto child = node->child(i);
            auto score = all_outputs[start + static_cast<size_t>(i)];
            auto state = NodeUnknown;

            if (score >= nn::NNScore(1.0)) {
                score = nn::NNScore(0.8999);
            } else if (score <= nn::NNScore(-1.0)) {
                score = nn::NNScore(-0.8999);
            }

            auto &pos = child->pos;
            if (pos.is_draw()) {
                score = nn::NNScore(0.0);
                state = NodeDraw;
            } else if (pos.is_lose()) {
                score = score_lose(child->ply);
                state = NodeLose;
            } else if (pos.is_win()) {
                score = score_win(child->ply);
                state = NodeWin;
            }
#if 1
            else if (!attack::in_checked(pos)) {
                const auto move = mate::mate_search(pos, 5);
                if (move != MOVE_NONE) {
                    score = score_win(child->ply);
                    state = NodeWin;
                }
            } else {
                const auto move = mate::mated_search(pos, 4);
                if (move == MOVE_NONE) {
                    score = score_lose(child->ply);
                    state = NodeLose;
                }
            }
#endif
            child->w = child->init_w = score;
            child->state = state;
        }
    }
}

void UBFMSearcherLocal::backup_vl(std::vector<VLLeafInfo> &leaves) {
    for (auto &info : leaves) {
        for (auto *node : info.path) {
            node->lock_node.lock();
            node->vl--;
            node->n++;
            node->lock_node.unlock();
        }
        for (int i = static_cast<int>(info.path.size()) - 1; i >= 0; i--) {
            auto *node = info.path[static_cast<size_t>(i)];
            node->lock_node.lock();
            if (node->child_len > 0) {
                this->update_node(node);
            }
            node->lock_node.unlock();
        }
    }
}

void test_ubfm() {
    // ファイル名
    std::string filename = "miss.txt";

    // ファイルを開く
    std::ifstream file(filename);

    // ファイルが正常に開かれたかを確認
    if (!file.is_open()) {
        std::cerr << "ファイルを開けませんでした: " << filename << std::endl;
        return ;
    }

    // ファイルから1行ずつ読み込む
    std::string line;
    while (std::getline(file, line)) {
        // ここで各行に対する処理を行う
        auto k = static_cast<Key>(to_u64(line));
        auto pos = hash::from_hash(k);
        ubfm::g_searcher_global.init();
        ubfm::think_ubfm(pos);
    }

    // ファイルを閉じる
    file.close();
}

}
#endif
