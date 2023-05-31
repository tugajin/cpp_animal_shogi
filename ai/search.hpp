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

Move search_root(game::Position &pos, int depth) {
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
    Tee<<best_score<<std::endl;
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
    if (pos.is_draw()) {
        return 0;
    }
    if (pos.is_lose()) {
        return -SEARCH_MATE + 10;
    }
    if (pos.is_win()) {
        return SEARCH_MATE - 10;
    }
    if (depth < 0) {
        return q_search(pos,alpha,beta);
        //return eval(pos);
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
    game::Position pos = hash::from_hash(153742799689785355);
    Tee<<pos<<std::endl;
    Tee<<move_str(search_root(pos,8))<<std::endl;
    // REP(i, INT_MAX) {
    //     game::Position pos;
    //     pos = hash::hirate();
    //     while(true) {
    //         Tee<<"自己対局："<<i<<std::endl;
    //         Tee<<pos<<std::endl;
    //         if (pos.is_done() || pos.is_draw_short()) {
    //             if (pos.is_draw() || pos.is_draw_short()) {Tee<<"引き分け\n";}
    //             if (pos.is_lose()) {Tee<<"負け\n";}
    //             break;
    //         }
    //         const auto best_move = search_root(pos,5);
    //         Tee<<move_str(best_move)<<std::endl;
    //         pos = pos.next(best_move);
    //     }
    // }
}

}
#endif