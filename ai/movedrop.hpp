#ifndef __MOVEDROP__HPP__
#define __MOVEDROP__HPP__

#include "game.hpp"
#include "movelist.hpp"
#include "attack.hpp"
#include <unordered_map>

namespace gen {

void drop_moves(const game::Position &pos, movelist::MoveList &ml) {
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
    if (!sp) { return; }
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto to = *p;
        const auto color_piece = pos.square(to);
        if (color_piece != COLOR_EMPTY) { continue; }
        REP(i, sp) {
            ml.add(move(to, hand_list[i]));
        }
    }
}

}
#endif