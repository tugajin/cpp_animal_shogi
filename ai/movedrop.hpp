#ifndef __MOVEDROP_HPP__
#define __MOVEDROP_HPP__

#include "game.hpp"
#include "movelist.hpp"
#include "attack.hpp"
#include <unordered_map>

namespace gen {

template<bool is_exists = false> bool drop_moves(const game::Position &pos, movelist::MoveList &ml) {
    const auto turn = pos.turn();
    const auto hand = pos.hand(turn);
    Piece hand_list[3] = {};
    int sp = 0;
    if (has_piece(hand, HIYOKO)) {
        hand_list[sp++] = HIYOKO;
    }
    if (has_piece(hand, ZOU)) {
        hand_list[sp++] = ZOU;
    }
    if (has_piece(hand, KIRIN)) {
        hand_list[sp++] = KIRIN;
    }
    if (!sp) { return false; }
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto to = *p;
        const auto color_piece = pos.square(to);
        if (color_piece != COLOR_EMPTY) { continue; }
        REP(i, sp) {
            if (is_exists) {
                return true;
            }
            ml.add(move(to, hand_list[i]));
        }
    }
    return false;
}

}
#endif