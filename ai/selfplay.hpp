#ifndef __SELFPLAY_HPP__
#define __SELFPLAY_HPP__

#include <vector>
#include <utility>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
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
#include "search.hpp"

#define DEBUG_OUT 0
static constexpr bool USE_DESCENT = false;

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
        // ファイル名の衝突防止: timestamp は秒精度、my_rand(9999) も狭いため、
        // 12 ワーカーが同時刻に同じ接尾辞を引くと .tmp を共有してしまい、
        // 片方の close() の rename が失敗して落ちる（2026-11-21 arm0 gen_2 で発生）。
        // プロセス内で単調増加するカウンタを付けて衝突自体を無くす。
        static std::atomic<int> file_counter{0};
        auto base = "./data/selfplay_" + timestamp() + "_" + to_string(my_rand(9999))
            + "_" + to_string(file_counter++);
        this->tmp_path = base + ".tmp";
        this->final_path = base + ".json";
        this->ofs.open(this->tmp_path);
    }
    void close() {
        this->ofs.close();
        std::filesystem::rename(this->tmp_path, this->final_path);
        this->info.clear();
    }
    void push_back(const Key hash, const nn::NNScore sc) {
        info.push_back({{"p", hash},{"s", sc},{"r", -99}});
    }
    void push_back(const Key hash, const nn::NNScore sc, const int cn) {
        info.push_back({{"p", hash},{"s", sc},{"r", -99},{"c", cn}});
    }
    void overwrite_result(double result) {
        if (USE_DESCENT) { return; }
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
    std::string tmp_path;
    std::string final_path;
};

class DescentSearcherLocal ;

class SelfPlayWorker : public ubfm::UBFMSearcherGlobal {
public:
#if DEBUG_OUT
    static constexpr int NUM = 1;
#else
    static constexpr int NUM = 12;
#endif
    void run_descent();
    void run_probe();
    void run_dump();
    void join();
    void init();
private:
    std::vector<DescentSearcherLocal> worker;
};

class DescentSearcherLocal : public ubfm::UBFMSearcherLocal {
public:
    DescentSearcherLocal(const int id, const int gpu_id, SelfPlayWorker * selfplay) :
    ubfm::UBFMSearcherLocal(id, gpu_id, selfplay) {
    }
    void run_descent();
    void run_probe();
    void run_dump();
    void search_descent(const uint32 simulation_num);
    void selfplay();
    void bishop_learning();
    ReplayBuffer replay_buffer;
    //reward::CountReward cw;
private:
    void evaluate_descent(ubfm::Node *node);
    void evaluate(ubfm::Node *node);
    Move execute_descent(game::Position &pos);
    void probe_loop();
    void dump_loop();
    ubfm::Node *probe_search(game::Position &pos);
    ubfm::Node *choice_best_move_probe();
    bool interrupt_descent(const uint32 current_num, const uint32 simulation_num) const;
    void add_replay_buffer(ubfm::Node *node);
    void choice_best_move_e_greedy();
    void choice_best_move_count();
    void choice_best_move_ordinal();
    //void choice_best_move_diff();
    void choice_best_move_softmax();
    void choice_best_move_random();
    void choice_best_move_greedy();
    //void choice_best_move_dropout();
    int po_num = 100;
};

class SelfPlayInfo {
private:
    std::atomic<int> num;
    int win_num;
    int lose_num;
    int draw_num;
    uint64 sum_ply;
    int num_limit;
    Lockable lock;
public:
    void init() {
        win_num = lose_num = draw_num = 0;
        num = num_limit = 0;
        sum_ply = 0;
    }
    void inc(const int result, const int ply) {
        lock.lock();
        num++;
        sum_ply += ply;
        switch(result) {
            case 0:
                draw_num++;
                break;
            case 1:
                win_num++;
                break;
            case -1:
                lose_num++;
                break;
        }
        lock.unlock();
    }
    double win_rate() const {
        double rate = (double(win_num) + (double(draw_num)/2)) / (double(win_num) + double(lose_num) + double(draw_num));
        return rate;
    }
    int sum_num() const {
        return win_num + lose_num + draw_num;
    }
    double avg_ply() const {
        return double(sum_ply)/double(this->sum_num());
    }
    std::string str() {
        std::string ret = "------------selfplay info [" + timestamp() + "]------------\n";
        const auto rate = this->win_rate();
        ret += "win:" + to_string(win_num)
            + " lose:" + to_string(lose_num)
            + " draw:" + to_string(draw_num)
            + " sum:" + to_string(this->sum_num())
            + " rate:" + to_string(rate)
            + " avg_ply:" + to_string(this->avg_ply());
        ret += "\n";
        return ret;
    }
    void set_limit(const int n) {
        num_limit = n;
    }
    bool is_end() const {
        return num >= num_limit;
    }
    void dump() const {
        std::ofstream ofs( "selfplay_result.csv", std::ios::out);
        ofs<<"win_num,lose_num,draw_num,avg_ply,percent,loss,ans\n";
        ofs<<this->win_num<<","
           <<this->lose_num<<","
           <<this->draw_num<<","
           <<this->avg_ply()<<","
           <<this->win_rate()<<","
           <<""<<","
           <<""<<"\n"
           ;
    }
};

extern SelfPlayWorker g_selfplay_worker[SelfPlayWorker::NUM];
extern int g_thread_counter;
extern SelfPlayInfo g_selfplay_info;

// probe モード（学習なしの探索測定、2026-11-22 追加）:
// 固定局面リストに対して生成時と同じ探索（po_num=100）を ε=0 で回し、
// root 値と選択した子局面のキーを返す。モデルを差し替えて同じリストを
// 測ることで「モデル＋探索」の出力を oracle と比較できる。
// g_probe_items は全ワーカーで共有するが、各ワーカーは自分の担当
// インデックス（thread_id 起点のストライド）だけを書くのでロックは不要。
struct ProbeItem {
    Key key = 0;
    double root_w = 0.0;
    Key child_key = 0;
    int ok = 0;
};
extern std::vector<ProbeItem> g_probe_items;
extern int g_probe_po;
void execute_probe(const std::string &keys_path, const int po_num);

// dump モード（予算の蒸留用、2026-11-23 追加）:
// 固定局面リストに対して浅い探索（po_num 回）を回し、探索木そのものを
// テキストで書き出す。学習側（train_distill.py）は木の葉を学習中モデルで
// 評価し直し、backup した root 値を深い探索の教師に合わせる。
// ノードの値（w）は update_node の規則により「子の符号反転の最大」と
// 常に一致する（解決済みの子がいても、値だけ見れば plain な negamax max
// になることを ubfm.hpp の update_node で確認済み）。そのため学習側は
// state を見ずに max だけで backup を再現できる。検証は train_distill.py
// の --verify（ダンプした葉の値を入れると root_w が再現されること）。
struct DumpItem {
    Key key = 0;
    std::string text;
    int ok = 0;
};
extern std::vector<DumpItem> g_dump_items;
extern int g_dump_po;
void execute_dump(const std::string &keys_path, const int po_num,
                  const std::string &out_path);

game::Position generate_random_position() {
    while (1) {
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY
            };
        Hand hand[COLOR_SIZE] = {HAND_NONE, HAND_NONE};
        Color turn = BLACK;
        Piece pieces_list[] = {HIYOKO, KIRIN, ZOU, HIYOKO, KIRIN, ZOU };
        
        const auto white_lion_index = my_rand(SQUARE_SIZE);
        pieces[white_lion_index] = WHITE_LION;
        auto black_lion_index = -1;
        while (1) {
            black_lion_index = my_rand(SQUARE_SIZE);
            if (pieces[black_lion_index] == COLOR_EMPTY) {
                pieces[black_lion_index] = BLACK_LION;
                break;
            }
        }

        for (const auto pc : pieces_list) {
            while (1) {
                const auto index = my_rand(SQUARE_SIZE + 1);
                const auto which = my_rand(2);
                const auto color = (which == 0) ? BLACK : WHITE;
                const auto cp = color_piece(pc, color);
                if (index == SQUARE_SIZE) {
                    if (pc == LION) { continue; }
                    hand[color] = inc_hand(hand[color],pc);
                    break;
                } else if (pieces[index] == COLOR_EMPTY) {
                    if (pc != HIYOKO) {
                        pieces[index] = cp;
                        break;
                    } 
                    if (rand_double() < 0.5) {
                        pieces[index] = cp; 
                        break;
                    }
                    if (cp == BLACK_HIYOKO) {
                        pieces[index] = BLACK_NIWATORI;
                    } else {
                        pieces[index] = WHITE_NIWATORI;
                    }
                    break;
                }
            }
        }
        auto pos = game::Position(pieces, hand, turn);
        auto att = attack::Attacker(pos);
        if (att.attacker_num[BLACK][white_lion_index] > 1) {
            continue;
        }
        if (att.attacker_num[WHITE][black_lion_index] > 1) {
            continue;
        }
        if (pos.is_win() || pos.is_lose()) {
            continue;
        }
        return pos;
    }
}

// 実行時の探索パラメータを1行で標準出力する（Tee 経由で log.log にも残る）。
// 生成データの出自（どの設定で作られたか）を後からログで検証できるようにするため、
// execute_selfplay の開始時に必ず出す。grep の目印は行頭の "PARAMS "。
// 参照できる定数は直接参照し、コード中にリテラルで散っているものは出典を併記する。
// リテラル側を変更したらこの行も合わせること。
void print_search_params(const int num) {
    Tee<<"PARAMS"
       <<" games="<<num
       <<" workers="<<SelfPlayWorker::NUM
       <<" use_descent="<<(USE_DESCENT ? 1 : 0)
       <<" cn_search="<<(ubfm::g_use_cn_search ? 1 : 0)
       <<" batch_predict="<<(ubfm::g_use_batch_predict ? 1 : 0)
       <<" po_num=100"          // DescentSearcherLocal::po_num（selfplay() で毎手 100 を設定）
       <<" epsilon=0.1"          // choice_best_move_e_greedy の rand_double() < 0.1
       <<" random_start=0.5"     // selfplay() の rand_double() < 0.5（ランダム局面 / 平手）
       <<" vl_batch=8"           // ubfm::UBFMSearcherLocal::VL_BATCH_SIZE（protected のためリテラルで併記）
       <<" vl_weight=0.5"        // ubfm::UBFMSearcherLocal::VL_WEIGHT（同上）
       <<" vl_noise_sigma=0.2"   // ubfm.hpp next_child_vl / select_leaf_vl の rand_gaussian(0, 0.2)
       <<" mate_depth=5"         // ubfm.hpp evaluate / predict_batch_leaves の mate_search(pos, 5)
       <<" mated_depth=4"        // 同 mated_search(pos, 4)
       <<" cn_tau="<<ubfm::CN_TAU
       <<" score_win=0.9999-ply/2000"  // ubfm::score_win
       <<" draw_rule=rep4"       // selfplay() の pos.is_draw(4)
       <<"\n";
}

void execute_selfplay(const int num) {
    g_thread_counter = 0;
    g_selfplay_info.init();
    g_selfplay_info.set_limit(num);

    ubfm::g_use_batch_predict = true;
    print_search_params(num);
    model::g_batch_predictor.start(0, 128, 1000);

    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].init();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].run_descent();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].join();
    }

    model::g_batch_predictor.stop();
    ubfm::g_use_batch_predict = false;

    Tee<<g_selfplay_info.str();
    // auto &s = model::g_gpu_model[0].stats;
    // Tee<<"predict calls:"<<s.call_count.load()
    //    <<" total_batch:"<<s.total_batch.load()
    //    <<" avg_batch:"<<(s.call_count.load()>0 ? s.total_batch.load()/s.call_count.load() : 0)
    //    <<" total_time_us:"<<s.total_time_us.load()
    //    <<" avg_time_us:"<<(s.call_count.load()>0 ? s.total_time_us.load()/s.call_count.load() : 0)
    //    <<"\n";
}

void SelfPlayWorker::run_probe() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].run_probe();
}

void DescentSearcherLocal::run_probe() {
    this->thread = new std::thread([this]() {
        this->probe_loop();
    });
}

void DescentSearcherLocal::probe_loop() {
    for (auto idx = static_cast<size_t>(this->thread_id);
         idx < g_probe_items.size();
         idx += SelfPlayWorker::NUM) {
        auto &item = g_probe_items[idx];
        auto pos = hash::from_hash(item.key);
        this->global->clear_tree();
        this->po_num = g_probe_po;
        auto child = this->probe_search(pos);
        if (child == nullptr) {
            item.ok = 0;
            continue;
        }
        item.root_w = this->root_node()->w;
        item.child_key = hash::hash_key(child->pos);
        item.ok = 1;
    }
}

ubfm::Node *DescentSearcherLocal::probe_search(game::Position &pos) {
    // execute_descent と同じ手順（root セット → search_descent）だが、
    // 手の選択だけ ε なしの probe 版に替える。
    this->root_node()->pos = pos;
    this->search_descent(this->po_num);
    if (this->root_node()->child_len == 0) {
        return nullptr;
    }
    return this->choice_best_move_probe();
}

ubfm::Node *DescentSearcherLocal::choice_best_move_probe() {
    // choice_best_move_e_greedy と同じスコア（解決済み子を優先し、
    // それ以外は -child->w）から、ε グリーディ（10% ランダム）と
    // タイブレーク用の微小乱数（rand_double()/1000）を除いた決定的 argmax。
    // probe は「探索が本番で選ぶ手」を測るので、探索的な手は入れない。
    std::vector<nn::NNScore> scores;
    REP(i, this->root_node()->child_len) {
        scores.push_back(nn::NNScore(0.0));
    }
    REP(i, this->root_node()->child_len) {
        auto child = this->root_node()->child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(j, this->root_node()->child_len) {
                    scores[j] = nn::NNScore(0.0);
                }
                scores[i] = ubfm::score_win(0);
                break;
            } else if (child->is_draw()) {
                scores[i] = nn::NNScore(0.0);
            } else if (child->is_win()) {
                scores[i] = ubfm::score_lose(0);
            }
        } else {
            scores[i] = -child->w;
        }
    }
    auto iter = std::max_element(scores.begin(), scores.end());
    auto index = std::distance(scores.begin(), iter);
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
    return child;
}

void execute_probe(const std::string &keys_path, const int po_num) {
    // キーファイル: 1行1キー（uint64 の10進表記）。空行と # 始まりは無視。
    std::ifstream ifs(keys_path);
    if (!ifs) {
        Tee<<"probe: キーファイルを開けません: "<<keys_path<<"\n";
        return;
    }
    g_probe_po = po_num;
    g_probe_items.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        ProbeItem item;
        item.key = std::stoull(line);
        g_probe_items.push_back(item);
    }
    g_thread_counter = 0;

    ubfm::g_use_batch_predict = true;
    Tee<<"PROBE_PARAMS"
       <<" positions="<<g_probe_items.size()
       <<" workers="<<SelfPlayWorker::NUM
       <<" use_descent="<<(USE_DESCENT ? 1 : 0)
       <<" cn_search="<<(ubfm::g_use_cn_search ? 1 : 0)
       <<" batch_predict=1"
       <<" po_num="<<g_probe_po
       <<" epsilon=0"
       <<"\n";
    model::g_batch_predictor.start(0, 128, 1000);

    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].init();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].run_probe();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].join();
    }

    model::g_batch_predictor.stop();
    ubfm::g_use_batch_predict = false;

    for (const auto &item : g_probe_items) {
        if (item.ok) {
            std::ostringstream oss;
            oss<<"PROBE "<<item.key
               <<" "<<std::fixed<<std::setprecision(6)<<item.root_w
               <<" "<<item.child_key<<"\n";
            Tee<<oss.str();
        } else {
            Tee<<"PROBE_SKIP "<<item.key<<"\n";
        }
    }
}

void SelfPlayWorker::run_dump() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].run_dump();
}

void DescentSearcherLocal::run_dump() {
    this->thread = new std::thread([this]() {
        this->dump_loop();
    });
}

// 木を preorder で集める（index 0 が root）。
inline void collect_nodes(ubfm::Node *node, std::vector<ubfm::Node*> &out) {
    out.push_back(node);
    if (node->child_len > 0) {
        REP(i, node->child_len) {
            collect_nodes(node->child(i), out);
        }
    }
}

// 木のテキスト化: TREE 行（root キーと root の w）に続き、ノードごとに
// 「N <idx> <key> <state> <w> <子数> <子 idx...>」、最後に END。
// state は ubfm::NodeState の int 値（Draw=0 / Win=1 / Lose=2 / Unknown=-1）。
inline std::string serialize_tree(ubfm::Node *root) {
    std::vector<ubfm::Node*> nodes;
    collect_nodes(root, nodes);
    std::unordered_map<ubfm::Node*, int> index;
    REP(i, static_cast<int>(nodes.size())) {
        index[nodes[i]] = i;
    }
    std::ostringstream oss;
    oss<<"TREE "<<hash::hash_key(root->pos)
       <<" "<<std::fixed<<std::setprecision(9)<<static_cast<double>(root->w)<<"\n";
    REP(i, static_cast<int>(nodes.size())) {
        auto *node = nodes[i];
        const auto n_child = std::max(node->child_len, 0);
        oss<<"N "<<i
           <<" "<<hash::hash_key(node->pos)
           <<" "<<static_cast<int>(node->state)
           <<" "<<std::fixed<<std::setprecision(9)<<static_cast<double>(node->w)
           <<" "<<n_child;
        REP(j, n_child) {
            oss<<" "<<index[node->child(j)];
        }
        oss<<"\n";
    }
    oss<<"END\n";
    return oss.str();
}

void DescentSearcherLocal::dump_loop() {
    for (auto idx = static_cast<size_t>(this->thread_id);
         idx < g_dump_items.size();
         idx += SelfPlayWorker::NUM) {
        auto &item = g_dump_items[idx];
        auto pos = hash::from_hash(item.key);
        this->global->clear_tree();
        this->po_num = g_dump_po;
        this->root_node()->pos = pos;
        this->search_descent(this->po_num);
        if (this->root_node()->child_len <= 0) {
            item.ok = 0;
            continue;
        }
        item.text = serialize_tree(this->root_node());
        item.ok = 1;
    }
}

void execute_dump(const std::string &keys_path, const int po_num,
                  const std::string &out_path) {
    std::ifstream ifs(keys_path);
    if (!ifs) {
        Tee<<"dump: キーファイルを開けません: "<<keys_path<<"\n";
        return;
    }
    g_dump_po = po_num;
    g_dump_items.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        DumpItem item;
        item.key = std::stoull(line);
        g_dump_items.push_back(item);
    }
    g_thread_counter = 0;

    ubfm::g_use_batch_predict = true;
    Tee<<"DUMP_PARAMS"
       <<" positions="<<g_dump_items.size()
       <<" workers="<<SelfPlayWorker::NUM
       <<" use_descent="<<(USE_DESCENT ? 1 : 0)
       <<" cn_search="<<(ubfm::g_use_cn_search ? 1 : 0)
       <<" batch_predict=1"
       <<" po_num="<<g_dump_po
       <<"\n";
    model::g_batch_predictor.start(0, 128, 1000);

    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].init();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].run_dump();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].join();
    }

    model::g_batch_predictor.stop();
    ubfm::g_use_batch_predict = false;

    std::ofstream ofs(out_path);
    if (!ofs) {
        Tee<<"dump: 出力ファイルを開けません: "<<out_path<<"\n";
        return;
    }
    ofs<<"DUMP_PARAMS positions="<<g_dump_items.size()
       <<" po_num="<<g_dump_po
       <<" workers="<<SelfPlayWorker::NUM<<"\n";
    auto n_ok = 0;
    for (const auto &item : g_dump_items) {
        if (item.ok) {
            ofs<<item.text;
            n_ok++;
        } else {
            ofs<<"TREE_SKIP "<<item.key<<"\n";
        }
    }
    Tee<<"dump: "<<n_ok<<"/"<<g_dump_items.size()
       <<" 木を "<<out_path<<" に書きました\n";
}

void SelfPlayWorker::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    this->worker.emplace_back(g_thread_counter,0,this);
    this->clear_tree();
    g_thread_counter++;
}

void SelfPlayWorker::run_descent() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].run_descent();
}

void SelfPlayWorker::join() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].join();
}

void DescentSearcherLocal::run_descent() {
    this->thread = new std::thread([this]() {
        this->selfplay();
        //this->bishop_learning();
    });
}
void DescentSearcherLocal::search_descent(const uint32 simulation_num) {
    for (auto i = 0u;;) {
        if (this->interrupt_descent(i, simulation_num)) break;

        // CN 探索時はバッチ前に木全体の共謀数を計算し直す（木は数百ノード規模なので安い）。
        // 根の CN が CN_INF（証明済み級）の間は共謀が存在しないため通常選択に任せる。
        auto cn_ok = false;
        if (ubfm::g_use_cn_search) {
            ubfm::compute_cn(this->root_node());
            cn_ok = ubfm::root_category_cn(this->root_node()) < ubfm::CN_INF;
        }
        std::vector<ubfm::VLLeafInfo> leaves;
        for (int b = 0; b < VL_BATCH_SIZE; b++) {
            if (this->root_node()->is_resolved()) break;
            if (cn_ok) {
                auto info = this->select_leaf_cn(this->root_node());
                if (info.leaf == nullptr) {
                    info = this->select_leaf_vl(this->root_node());
                }
                leaves.push_back(info);
            } else {
                leaves.push_back(this->select_leaf_vl(this->root_node()));
            }
        }
        if (leaves.empty()) break;

        this->predict_batch_leaves(leaves);
        this->backup_vl(leaves);
        i += static_cast<uint32>(leaves.size());
    }
}

void DescentSearcherLocal::evaluate_descent(ubfm::Node *node) {
    node->n++;
    ASSERT2(node->pos.is_ok(),{
        Tee<<node->pos<<std::endl;
    })
    ASSERT(std::fabs(node->w) <= 1);
    if (node->pos.is_draw()) {
        node->w = nn::NNScore(0.0);
        node->state = ubfm::NodeState::NodeDraw;
        return;
    }
    if (node->pos.is_lose()) {
        node->w = ubfm::score_lose(node->ply);
        node->state = ubfm::NodeState::NodeLose;
        return;
    }
    if (node->is_resolved()) {
        return;
    }
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
        this->update_node(node);
    }
    auto next_node = this->next_child(node);
    if (next_node != nullptr) {
        this->evaluate_descent(next_node);
        this->update_node(node);
    }
}
void DescentSearcherLocal::evaluate(ubfm::Node *node) {
    node->n++;
    ASSERT2(node->pos.is_ok(),{
        Tee<<node->pos<<std::endl;
    })
    ASSERT(std::fabs(node->w) <= 1);

    if (node->pos.is_draw()) {
        node->w = nn::NNScore(0.0);
        node->state = ubfm::NodeState::NodeDraw;
        return;
    }
    if (node->pos.is_lose()) {
        node->w = ubfm::score_lose(node->ply);
        node->state = ubfm::NodeState::NodeLose;
        return;
    }
    if (node->is_resolved()) {
        return;
    }
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
    } else {
        auto next_node = this->next_child(node);
        if (next_node == nullptr) {
            return;
        }
        this->evaluate(next_node);
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

void DescentSearcherLocal::add_replay_buffer(ubfm::Node *node) {
    const auto key = hash::hash_key(node->pos);
    this->replay_buffer.push_back(key, node->w);
    REP(i, node->child_len) {

        ASSERT(i>=0);
        ASSERT(i<node->child_len);
        auto child = node->child(i);

        if (child->is_terminal() && !child->is_resolved()) {
            continue;
        }
        this->add_replay_buffer(child);
    }
}

Move DescentSearcherLocal::execute_descent(game::Position &pos) {
    this->root_node()->pos = pos;
    this->search_descent(this->po_num);
    this->choice_best_move_e_greedy();
    //this->choice_best_move_count();
    //this->choice_best_move_ordinal();
    //this->choice_best_move_diff();
    //this->choice_best_move_softmax();
    //this->choice_best_move_random();
    //this->choice_best_move_greedy();
    //this->choice_best_move_dropout();
    if (USE_DESCENT) {
        this->add_replay_buffer(this->root_node());
    } else {
        const auto k = hash::hash_key(this->root_node()->pos);
        const auto w = this->root_node()->w;
        if (ubfm::g_use_cn_search) {
            // 最終的な木から根のカテゴリ共謀数を注釈する（CN_INF は -1 で記録 = 証明済み級）
            ubfm::compute_cn(this->root_node());
            const auto cn = ubfm::root_category_cn(this->root_node());
            this->replay_buffer.push_back(k, w, cn >= ubfm::CN_INF ? -1 : cn);
        } else {
            this->replay_buffer.push_back(k,w);
        }
    }
   return this->root_node()->best_move;
}

void DescentSearcherLocal::choice_best_move_e_greedy() {

    std::vector<nn::NNScore> scores;
    REP(i, this->root_node()->child_len) {
        scores.push_back(nn::NNScore(0.0));
    }
    auto find_resolved_flag = false;
    REP(i, this->root_node()->child_len) {
        auto child = this->root_node()->child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = nn::NNScore(0.0);
                }
                scores[i] = ubfm::score_win(0);
                find_resolved_flag = true;
                break;
            } else if (child->is_draw()) {
                scores[i] = nn::NNScore(0.0) + (rand_double() / 1000);
            } else if (child->is_win()) {
                scores[i] = ubfm::score_lose(0) + (rand_double() / 1000);
            }
        } else {
            scores[i] = -child->w;
        }
    }
    auto index = -1;
    if (!find_resolved_flag && rand_double() < 0.1) {
        index = my_rand(this->root_node()->child_len);
    } else {
        auto iter = std::max_element(scores.begin(), scores.end());
        index = std::distance(scores.begin(), iter);
    }
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
}

void DescentSearcherLocal::choice_best_move_count() {
    std::vector<double> scores;
    std::vector<uint64> num;
    REP(i, this->root_node()->child_len) {

        ASSERT(i>=0);
        ASSERT(i<this->root_node()->child_len);
        auto child = this->root_node()->child(i);
        const auto r = 1; //+ cw.get(child->pos.history());
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
#if DEBUG_OUT
        Tee<<"n:"<<padding_str(to_string(child->n),3) 
            << " w:" << padding_str(to_string(-child->w),7) 
          //  << " oracle:" << padding_str(to_string(oracle),2) 
            << " org:" << padding_str(to_string(num[i]),3)<<" "
            << " score:" << padding_str(to_string(scores[i]),7)<<" "
            <<move_str(child->parent_move)<<std::endl;
#endif
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
void DescentSearcherLocal::choice_best_move_ordinal() {

    std::vector<nn::NNScore> scores;
    REP(i, this->root_node()->child_len) {
        scores.push_back(nn::NNScore(0.0));
    }
    auto find_resolved_flag = false;
    REP(i, this->root_node()->child_len) {
        auto child = this->root_node()->child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = nn::NNScore(0.0);
                }
                scores[i] = ubfm::score_win(0);
                find_resolved_flag = true;
                break;
            } else if (child->is_draw()) {
                scores[i] = nn::NNScore(0.0) + (rand_double() / 1000);
            } else if (child->is_win()) {
                scores[i] = ubfm::score_lose(0) + (rand_double() / 1000);
            }
        } else {
            scores[i] = static_cast<nn::NNScore>(child->n) + (rand_double() / 10);
        }
    }
    auto index = -1;
#if DEBUG_OUT
    Tee<<"\n";
    REP(i,scores.size()) {
        Tee<<i<<":"<<scores[i]<<std::endl;
    }
#endif
    if (!find_resolved_flag) {
        double prob = 1;
        while(prob > 0.7) {
            auto iter = std::max_element(scores.begin(), scores.end());
            index = std::distance(scores.begin(), iter);
            scores[index] = static_cast<nn::NNScore>(-1);
#if DEBUG_OUT
            Tee<<"select:"<<index<<" ";
#endif
            prob = rand_double();
        }
    } else {
        auto iter = std::max_element(scores.begin(), scores.end());
        index = std::distance(scores.begin(), iter);
    }
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;

}

// void DescentSearcherLocal::choice_best_move_diff() {
// }

void DescentSearcherLocal::choice_best_move_softmax() {
    std::vector<nn::NNScore> scores;
    REP(i, this->root_node()->child_len) {
        scores.push_back(nn::NNScore(0.0));
    }
    auto find_resolved_flag = false;
    auto max_score = nn::NNScore(-1);
    REP(i, this->root_node()->child_len) {
        auto child = this->root_node()->child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = nn::NNScore(0.0);
                }
                scores[i] = ubfm::score_win(0);
                find_resolved_flag = true;
                break;
            } else if (child->is_draw()) {
                scores[i] = child->n;
            } else if (child->is_win()) {
                scores[i] = 0;
            }
        } else {
            scores[i] = child->n;
        }
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }
    for(auto i = 0u; i < scores.size(); i++) {
        scores[i] = std::exp((scores[i] - max_score)/2);
#if DEBUG_OUT
        Tee<<scores[i]<<std::endl;
#endif
    }
    auto index = -1;
    if (!find_resolved_flag ) {
        index = my_choice(scores);
#if DEBUG_OUT
        Tee<<"select:"<<index<<std::endl;
#endif
    } else {
        auto iter = std::max_element(scores.begin(), scores.end());
        index = std::distance(scores.begin(), iter);
    }
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
}

void DescentSearcherLocal::choice_best_move_random() {
    std::vector<nn::NNScore> scores;
    REP(i, this->root_node()->child_len) {
        scores.push_back(nn::NNScore(0.0));
    }
    REP(i, this->root_node()->child_len) {
        auto child = this->root_node()->child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = nn::NNScore(0.0);
                }
                scores[i] = ubfm::score_win(0);
                break;
            } else if (child->is_draw()) {
                scores[i] = rand_double();
            } else if (child->is_win()) {
                scores[i] = rand_double()/1000;
            }
        } else {
            scores[i] = rand_double();
        }
    }
    auto iter = std::max_element(scores.begin(), scores.end());
    auto index = std::distance(scores.begin(), iter);
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
}

void DescentSearcherLocal::choice_best_move_greedy() {

    std::vector<nn::NNScore> scores;
    REP(i, this->root_node()->child_len) {
        scores.push_back(nn::NNScore(0.0));
    }
    REP(i, this->root_node()->child_len) {
        auto child = this->root_node()->child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = nn::NNScore(0.0);
                }
                scores[i] = ubfm::score_win(0);
                break;
            } else if (child->is_draw()) {
                scores[i] = child->n + (rand_double() / 1000);
            } else if (child->is_win()) {
                scores[i] = 0;
            }
        } else {
            scores[i] = child->n + (rand_double() / 1000);
        }
    }
    auto iter = std::max_element(scores.begin(), scores.end());
    auto index = std::distance(scores.begin(), iter);
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
}

// void DescentSearcherLocal::choice_best_move_dropout() {
// }

void DescentSearcherLocal::selfplay() {
    Tee<<"start selfplay\n";
    for(auto i = 0; !g_selfplay_info.is_end() ; i++) {
        game::Position pos;
        if (rand_double() < 0.5) {
            pos = generate_random_position();
        } else {
            pos = hash::hirate();
        }
        this->replay_buffer.open();
        while(true) {
#if DEBUG_OUT
            Tee<<"自己対局("<<this->thread_id<<")："<<i<<":"<<pos.ply()<<std::endl;
            Tee<<pos<<std::endl;
#endif
            this->global->clear_tree();

            if (pos.is_lose() || pos.is_draw(4)) {
                auto result = 0.0;
                if (pos.is_draw(4)) {
                    result = 0.0;
                    g_selfplay_info.inc(0,pos.ply());
                }
                if (pos.is_lose()) {
                    if (pos.turn() == BLACK) {
                        result = -1.0;
                        g_selfplay_info.inc(-1,pos.ply());
                    } else {
                        result = 1.0;
                        g_selfplay_info.inc(1,pos.ply());
                    }
                }
                this->replay_buffer.overwrite_result(result);
                this->replay_buffer.write_data();
                this->replay_buffer.close();
                break;
            }
            this->po_num = 100;

            auto best_move = execute_descent(pos);

            //this->cw.update(pos.history());

            // if (!attack::in_checked(pos)) {
            //     const auto mate_move = mate::mate_search(pos,5);
            //     if (mate_move != MOVE_NONE) {
            //         best_move = mate_move;
            //     }
            // }
            pos = pos.next(best_move);
        }
        if (i % 100 == 0) {
            //this->cw.dump();
        }
        if (this->thread_id == 0 && i % 100 == 0) {
            Tee<<g_selfplay_info.str();
        }
    }
    g_selfplay_info.dump();
}
void DescentSearcherLocal::bishop_learning() {
    Tee<<"start bishop_learning\n";
    this->replay_buffer.open();

    for(auto i = 0; !g_selfplay_info.is_end()  ; i++) {
        game::Position pos;
        pos = generate_random_position();
#if DEBUG_OUT
        Tee<<"bishop learning("<<this->thread_id<<")："<<i<<std::endl;
        Tee<<pos<<std::endl;
        Tee<<g_selfplay_info.str();
#endif

        this->global->clear_tree();
        this->po_num = 1000;
        execute_descent(pos);
        if (this->thread_id == 0 && i % 1000 == 0) {
            Tee<<g_selfplay_info.str();
        }
        g_selfplay_info.inc(0,0);
    }
    this->replay_buffer.write_data();
    this->replay_buffer.close();
    g_selfplay_info.dump();
}
}
#endif
