#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include <climits>
#include "game.hpp"
#include "common.hpp"
#include "movelegal.hpp"

namespace search {

constexpr int SEARCH_MATE = 10000;
constexpr int SEARCH_MAX  = 20000;
constexpr int SEARCH_MIN  = -SEARCH_MAX;

constexpr int HIYOKO_SCORE = 100;
constexpr int KIRIN_SCORE = 300;
constexpr int ZOU_SCORE = 300;
constexpr int LION_SCORE = 5000;
constexpr int NIWATORI_SCORE = 500;

int eval(const game::Position &pos) {
    auto score = 0;
    for(auto *fp = SQUARE_INDEX; *fp != SQ_WALL; ++fp) {
        const auto color_piece = pos.square(*fp);
        switch(color_piece) {
            case BLACK_HIYOKO: score += HIYOKO_SCORE; break;
            case BLACK_KIRIN: score += KIRIN_SCORE; break;
            case BLACK_ZOU: score += ZOU_SCORE; break;
            case BLACK_LION: score += LION_SCORE; break;
            case BLACK_NIWATORI: score += NIWATORI_SCORE; break;
            case WHITE_HIYOKO: score -= HIYOKO_SCORE; break;
            case WHITE_KIRIN: score -= KIRIN_SCORE; break;
            case WHITE_ZOU: score -= ZOU_SCORE; break;
            case WHITE_LION: score -= LION_SCORE; break;
            case WHITE_NIWATORI: score -= NIWATORI_SCORE; break;
            default: break;
        }
    }
    score += num_piece(pos.hand(BLACK),HIYOKO) * HIYOKO_SCORE;
    score += num_piece(pos.hand(BLACK),KIRIN)  * KIRIN_SCORE;
    score += num_piece(pos.hand(BLACK),ZOU)    * ZOU_SCORE;
    score -= num_piece(pos.hand(WHITE),HIYOKO) * HIYOKO_SCORE;
    score -= num_piece(pos.hand(WHITE),KIRIN)  * KIRIN_SCORE;
    score -= num_piece(pos.hand(WHITE),ZOU)    * ZOU_SCORE;
    score += static_cast<int>(rand_gaussian(0, 10));
    return (pos.turn() == BLACK) ? score : -score;
}

int search(game::Position &pos, int alpha, int beta, int depth);

Move search_root(game::Position &pos, int depth, int &best_sc) {
    auto best_move = MOVE_NONE;
    auto best_score = -SEARCH_MATE;
    auto alpha = SEARCH_MIN;
    auto beta = SEARCH_MAX;
    movelist::MoveList ml;
    gen::legal_moves(pos,ml);

    for (const auto m : ml) {
        auto next_pos = pos.next(m);
        const auto score = -search(next_pos,-beta, -alpha, depth-1);
        if (score > best_score) {
            best_score = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
            }
        }
    }
    best_sc = best_score;
    return best_move;
}
int q_search(game::Position &pos, int alpha, int beta) {
    ASSERT2(pos.is_ok(),{
        Tee<<pos<<std::endl;
    });

    ASSERT(alpha < beta);

    if (pos.is_draw()) {
        return 0;
    }
    if (pos.is_lose()) {
        return -SEARCH_MATE + 10;
    }
    if (pos.is_win()) {
        return SEARCH_MATE - 10;
    }
    auto best_score = SEARCH_MIN;
    movelist::MoveList ml;
    
    if (attack::in_checked(pos)) {
        gen::evasion_moves(pos,ml);
    } else {
        best_score = eval(pos);
        if (best_score > alpha) {
            alpha = best_score;
            if (best_score >= beta) {
                return best_score;
            }
        }
        gen::cap_moves(pos, ml);
    }
    for (const auto m : ml) {
        auto next_pos = pos.next(m);
        const auto score = -q_search(next_pos, -beta, -alpha);
        if (score > best_score) {
            best_score = score;
            alpha = score;
            if (score >= beta) {
                return best_score;
            }
        }
    }
    if (best_score == SEARCH_MIN) {
        return -SEARCH_MATE + 7;
    }
    return best_score;
}

int search(game::Position &pos, int alpha, int beta, int depth) {
    ASSERT2(pos.is_ok(),{
        Tee<<pos<<std::endl;
    });
    ASSERT(alpha < beta);
    if (pos.is_win()) {
        return SEARCH_MATE - 10;
    }
    if (pos.is_lose()) {
        return -SEARCH_MATE + 10;
    }
    if (pos.is_draw()) {
        return 0;
    }
    if (depth < 0) {
        //return q_search(pos,alpha,beta);
        //return eval(pos);
        return 0;
    }
    if (!attack::in_checked(pos)) {
        const auto m = mate::mate_search(pos,5);
        if (m != MOVE_NONE) {
            return SEARCH_MATE - 4;
        }
    }
    movelist::MoveList ml;
    gen::legal_moves(pos, ml);
    auto best_score = SEARCH_MIN;
    for (const auto m : ml) {
        auto next_pos = pos.next(m);
        const auto score = -search(next_pos, -beta, -alpha, depth-1);
        if (score > best_score) {
            best_score = score;
            alpha = score;
            if (score >= beta) {
                return best_score;
            }
        }
    }
    if (best_score == SEARCH_MIN) {
        return -SEARCH_MATE + 7;
    }
    return best_score;
}
void test_search() {
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
        auto sc = 0;
        search_root(pos,10,sc);
        Tee<<pos<<std::endl;
        Tee<<"score:"<<sc<<std::endl;
    }

    // ファイルを閉じる
    file.close();
}

}
#endif