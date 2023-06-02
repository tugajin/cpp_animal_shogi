#ifndef __REVIEW_BUFFER_HPP__
#define __REVIEW_BUFFER_HPP__
#include "game.hpp"
#include "ubfm.hpp"
#include "movelist.hpp"
#include "movelegal.hpp"
#include <vector>
#include <utility>
#include <set>

namespace review {
class ReviewBuffer {
private:
    std::vector<std::pair<game::Position,int>> buffer;
    std::set<Key> black_list;
public:
    size_t size() const {
        return buffer.size();
    }
    bool find(const Key k) {
        for(auto &item : this->buffer) {
            const auto pos = item.first;
            const auto pos_key = hash::hash_key(pos);
            if (k == pos_key) {
                return true;
            }
        }
        return false;
    }
    void add(const ubfm::Node *node) {
        const auto k = hash::hash_key(node->pos);
        if (this->find(k)) { 
            Tee<<"追加済";
            return; 
        }
        // if (this->black_list.contains(k)) { 
        //     Tee<<"black";
        //     return; 
        // }
        const auto score = this->eval(node);
        if (score > 0) {
            Tee<<"追加"<<score<<"\n";
            this->buffer.push_back(std::make_pair(node->pos,0));
            //this->black_list.insert(node->pos.history());
        }
    }
    game::Position choice() {
        ASSERT(this->buffer.size() > 0);
        const auto index = my_rand(this->buffer.size());
        auto pair = buffer[index];
        pair.second = pair.second + 1;
        if (/*pair.second > 3*/true) {
            this->buffer.erase(this->buffer.cbegin()+index);
            //const auto pos = pair.first;
            //const auto k = hash::hash_key(pos);
            //this->black_list.insert(k);
        } else {
            this->buffer[index] = pair;
        }
        return pair.first;
    }
    int eval(const ubfm::Node *node) {
        nn::NNScore max_score = static_cast<nn::NNScore>(-2);
        std::vector<uint32> n_list;
        REP(i, node->child_len) {
            const auto child = node->child(i);
            if (-child->w > max_score) { max_score = -child->w; }
            auto n = child->n;
            if (-child->w < -0.9) {
                n = 0;
            }
            n_list.push_back(n);
        }
        if (std::fabs(max_score) > 0.4) {
            return -1;
        }
        std::sort(n_list.begin(),n_list.end());
        std::reverse(n_list.begin(),n_list.end());
        if (n_list.size() <= 1) {
            return 0;
        } else if (n_list.size() == 2) {
            if (n_list[0] - n_list[1] < 10) {
                return 1;
            }
        } else {
            if (n_list[1] + n_list[2] > n_list[0]) {
                return 1;
            }
        }
        return 0;
    }
};

extern ReviewBuffer g_review_buffer;

void test_review() {
    ubfm::Node node;
    node.pos = hash::hirate();
    auto moveList = movelist::MoveList();
    gen::legal_moves(node.pos, moveList);
    
    node.child_len = moveList.len();
    node.child_nodes = std::make_unique<std::unique_ptr<ubfm::Node>[]>(node.child_len);
    REP(i, node.child_len) {
        auto next_pos = node.pos.next(moveList[i]);
        node.child_nodes[i] = std::make_unique<ubfm::Node>();
        auto next_node = node.child_nodes[i].get();
        next_node->pos = next_pos;
        next_node->ply = node.ply+1;
        next_node->parent_move = moveList[i];
        next_node->w = static_cast<nn::NNScore>(rand_double());
    }
    REP(i, 100){
        REP(j, node.child_len) {
            auto child = node.child(j);
            child->w = static_cast<nn::NNScore>(rand_double());
        }
        g_review_buffer.add(&node);
    }
    while (g_review_buffer.size() > 0) {
        g_review_buffer.choice();
    }
    REP(i, 100){
        REP(j, node.child_len) {
            auto child = node.child(j);
            child->w = static_cast<nn::NNScore>(rand_double());
        }
        g_review_buffer.add(&node);
    }
}

}

#endif