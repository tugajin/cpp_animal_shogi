#ifndef __MOVECHECK__HPP__
#define __MOVECHECK__HPP__

#include "game.hpp"
#include "movelist.hpp"
#include "attack.hpp"
#include <unordered_map>

namespace gen {

void pos_check_moves(const game::Position &pos, movelist::MoveList &ml) {
    const auto me = pos.turn();
    const auto opp = change_turn(me);
    const auto opp_lion_sq = pos.king_sq(opp);
    const auto move_flag = (me == BLACK) ? (BLACK_FLAG | COLOR_WALL_FLAG) : (WHITE_FLAG | COLOR_WALL_FLAG);
    // HIYOKO
    {
        const auto size = pos.piece_list_size(me, HIYOKO);
        REP(index, size) {
            const auto from = pos.piece_list(me, HIYOKO, index);
            if (me == BLACK) {
                const auto to = from + INC_UP;
                const auto prom = sq_rank(to) == RANK_1;
                if (prom) { 
                    if (pseudo_attack(BLACK_NIWATORI, opp_lion_sq - to) && attack::can_move(pos.square(to), move_flag)) {
                        ml.add(move(from, to, true));
                    }
                } else {
                    if (to + INC_UP == opp_lion_sq && attack::can_move(pos.square(to), move_flag)) {
                        ml.add(move(from, to));
                    }
                }
            } else {
                const auto to = from + INC_DOWN;
                const auto prom = sq_rank(to) == RANK_4;
                if (prom) { 
                    if (pseudo_attack(WHITE_NIWATORI, opp_lion_sq - to) && attack::can_move(pos.square(to), move_flag)) {
                        ml.add(move(from, to, true));
                    }
                } else {
                    if (to + INC_DOWN == opp_lion_sq && attack::can_move(pos.square(to), move_flag)) {
                        ml.add(move(from, to));
                    }
                }
            }
        }
    }
#define ADD_MOVE(dir, piece) do {\
            const auto to = from + (dir);\
            if (pseudo_attack(piece, opp_lion_sq - to) && attack::can_move(pos.square(to), move_flag)) {\
                ml.add(move(from, to));\
            }\
}while(false)
    // ZOU
    {
        const auto size = pos.piece_list_size(me, ZOU);
        REP(index, size) {
            const auto from = pos.piece_list(me, ZOU, index);
            ADD_MOVE(INC_LEFTUP, BLACK_ZOU);//HACK ZOUは先手後手同じなので
            ADD_MOVE(INC_RIGHTUP, BLACK_ZOU);
            ADD_MOVE(INC_LEFTDOWN, BLACK_ZOU);
            ADD_MOVE(INC_RIGHTDOWN, BLACK_ZOU);
        }
    }
    // KIRIN
    {
        const auto size = pos.piece_list_size(me, KIRIN);
        REP(index, size) {
            const auto from = pos.piece_list(me, KIRIN, index);
            ADD_MOVE(INC_UP, BLACK_KIRIN);//HACK KIRINは先手後手同じなので
            ADD_MOVE(INC_DOWN, BLACK_KIRIN);
            ADD_MOVE(INC_LEFT, BLACK_KIRIN);
            ADD_MOVE(INC_RIGHT, BLACK_KIRIN);
        }
    }
    // NIWATORI
    {
        const auto size = pos.piece_list_size(me, NIWATORI);
        REP(index, size) {
            const auto from = pos.piece_list(me, NIWATORI, index);
            if (me == BLACK) {
                ADD_MOVE(INC_UP, BLACK_NIWATORI);
                ADD_MOVE(INC_DOWN, BLACK_NIWATORI);
                ADD_MOVE(INC_LEFT, BLACK_NIWATORI);
                ADD_MOVE(INC_RIGHT, BLACK_NIWATORI);
                ADD_MOVE(INC_LEFTUP ,BLACK_NIWATORI);
                ADD_MOVE(INC_RIGHTUP,BLACK_NIWATORI);
            } else {
                ADD_MOVE(INC_UP, WHITE_NIWATORI);
                ADD_MOVE(INC_DOWN, WHITE_NIWATORI);
                ADD_MOVE(INC_LEFT, WHITE_NIWATORI);
                ADD_MOVE(INC_RIGHT, WHITE_NIWATORI);
                ADD_MOVE(INC_LEFTDOWN,WHITE_NIWATORI);
                ADD_MOVE(INC_RIGHTDOWN,WHITE_NIWATORI);
            }
        }
    }

#undef ADD_MOVE
}

void drop_check_moves(const game::Position &pos, movelist::MoveList &ml) {
    const auto me = pos.turn();
    const auto opp = change_turn(me);
    const auto hand = pos.hand(me);
    const auto opp_lion_sq = pos.king_sq(opp);
    if (hand == HAND_NONE) { return; }
    if (has_piece(hand, HIYOKO)) {
        if (me == BLACK) {
            const auto to = opp_lion_sq + INC_DOWN;
            if (pos.square(to) == COLOR_EMPTY) {
                ml.add(move(to, HIYOKO));
            }
        } else {
            const auto to = opp_lion_sq + INC_UP;
            if (pos.square(to) == COLOR_EMPTY) {
                ml.add(move(to, HIYOKO));
            }    
        }
    }
#define ADD_MOVE(dir, piece) do {\
            const auto to = opp_lion_sq + (dir);\
            if (pos.square(to) == COLOR_EMPTY) {\
                ml.add(move((to), (piece)));\
            }\
}while(false)
    if (has_piece(hand, KIRIN)) {
        ADD_MOVE(INC_UP, KIRIN);
        ADD_MOVE(INC_DOWN, KIRIN);
        ADD_MOVE(INC_LEFT, KIRIN);
        ADD_MOVE(INC_RIGHT, KIRIN);
    }
    if (has_piece(hand, ZOU)) {
        ADD_MOVE(INC_LEFTUP, ZOU);
        ADD_MOVE(INC_LEFTDOWN, ZOU);
        ADD_MOVE(INC_RIGHTUP, ZOU);
        ADD_MOVE(INC_RIGHTDOWN, ZOU);
    }
#undef ADD_MOVE
}

void check_moves(const game::Position &pos, movelist::MoveList &ml) {
    drop_check_moves(pos,ml);
    pos_check_moves(pos,ml);
}
}
#endif