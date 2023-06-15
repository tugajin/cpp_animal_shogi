#ifndef __MATESEARCH_HPP__
#define __MATESEARCH_HPP__

#include "game.hpp"
#include "movecheck.hpp"
#include "moveevasion.hpp"
#include "attack.hpp"
#include "movelist.hpp"

namespace mate {

enum MateResult {
    MATED,
    NOT_MATED,
    UNKNOWN,
};
MateResult mate_and(const game::Position &pos, int ply);
MateResult mate_or(const game::Position &pos, int ply);

Move mate_search(const game::Position &pos, int ply) {
    ASSERT(!attack::in_checked(pos));
    movelist::MoveList ml;
    gen::check_moves(pos,ml);
    const auto len = ml.len();
    REP(i, len) {
        const auto next = pos.next(ml[i]);
        const auto result = mate_and(next, ply-1);
        if (result == MATED) {
            return ml[i];
        }
    }
    return MOVE_NONE;
}

Move mated_search(const game::Position &pos, int ply) {
    ASSERT(attack::in_checked(pos));
    movelist::MoveList ml;
    gen::evasion_moves(pos,ml);
    const auto len = ml.len();
    REP(i, len) {
        const auto next = pos.next(ml[i]);
        const auto result = mate_or(next, ply-1);
        if (result == NOT_MATED) {
            return ml[i];
        }
    }
    return MOVE_NONE;
}

MateResult mate_and(const game::Position &pos, int ply) {
    if (ply < 0) {
        return NOT_MATED;
    }
    if (pos.is_draw()) {
        return NOT_MATED;
    }
    if (pos.is_lose()) {
        return MATED;
    }
    ASSERT(attack::in_checked(pos));
    movelist::MoveList ml;
    gen::evasion_moves(pos, ml);
    const auto len = ml.len();
    REP(i, len) {
        const auto next = pos.next(ml[i]);
        const auto result = mate_or(next, ply-1);
        if (result == NOT_MATED) {
            return NOT_MATED;
        }
    }
    return MATED;
}

MateResult mate_or(const game::Position &pos, int ply) {
    if (ply < 0) {
        return NOT_MATED;
    }
    if (pos.is_draw()) {
        return NOT_MATED;
    }
    if (pos.is_lose()) {
        return NOT_MATED;
    }
    if (attack::in_checked(pos)) {
        return NOT_MATED;
    }
    movelist::MoveList ml;
    gen::check_moves(pos, ml);
    const auto len = ml.len();
    REP(i, len) {
        const auto next = pos.next(ml[i]);
        const auto result = mate_and(next, ply-1);
        if (result == MATED) {
            return MATED;
        }
    }
    return NOT_MATED;
}

void test_mate() {
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            CE, WL, CE,
            CE, CE, BK,
            CE, BL, CE,
            CE, CE, CE,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        hand[BLACK] = inc_hand(hand[BLACK],KIRIN);
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        Tee<<move_str(mate_search(pos,3))<<std::endl;
    }
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            CE, WL, CE,
            CE, CE, CE,
            CE, BL, CE,
            CE, CE, CE,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        Tee<<move_str(mate_search(pos,3))<<std::endl;
    }
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            CE, WL, CE,
            CE, CE, CE,
            BL, CE, BN,
            CE, CE, CE,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        Tee<<move_str(mate_search(pos,3))<<std::endl;
    }
}

}
#endif