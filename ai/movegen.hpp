#ifndef __MOVEGEN__HPP__
#define __MOVEGEN_HPP__

#include "game.hpp"
#include "movelist.hpp"
#include "attack.hpp"
#include <unordered_map>

namespace gen {

void pos_moves(const game::Position &pos, movelist::MoveList &ml) {
    const auto turn = pos.turn();
    const auto opp = change_turn(turn);
    const auto move_flag = (turn == BLACK) ? (BLACK_FLAG | COLOR_WALL_FLAG) : (WHITE_FLAG | COLOR_WALL_FLAG);
    // HIYOKO
    {
        const auto size = pos.piece_list_size(turn, HIYOKO);
        REP(index, size) {
            const auto from = pos.piece_list(turn, HIYOKO, index);
            if (turn == BLACK) {
                const auto to = from + INC_UP;
                if (attack::can_move(pos.square(to), move_flag)) {
                    const auto prom = sq_rank(to) == RANK_1;
                    if (prom) {
                        ml.add(move(from, to, prom));
                    }
                    ml.add(move(from, to));
                }
            } else {
                const auto to = from + INC_DOWN;
                if (attack::can_move(pos.square(to), move_flag)) {
                    const auto prom = sq_rank(to) == RANK_4;
                    if (prom) {
                        ml.add(move(from, to, prom));
                    }
                    ml.add(move(from, to));
                }
            }
        }
    }
#define ADD_MOVE(dir) do {\
            const auto to = from + (dir);\
            if (attack::can_move(pos.square(to), move_flag)) {\
                ml.add(move(from, to));\
            }\
}while(false)
    // ZOU
    {
        const auto size = pos.piece_list_size(turn, ZOU);
        REP(index, size) {
            const auto from = pos.piece_list(turn, ZOU, index);
            ADD_MOVE(INC_LEFTUP);
            ADD_MOVE(INC_RIGHTUP);
            ADD_MOVE(INC_LEFTDOWN);
            ADD_MOVE(INC_RIGHTDOWN);
        }
    }
    // KIRIN
    {
        const auto size = pos.piece_list_size(turn, KIRIN);
        REP(index, size) {
            const auto from = pos.piece_list(turn, KIRIN, index);
            ADD_MOVE(INC_UP);
            ADD_MOVE(INC_DOWN);
            ADD_MOVE(INC_LEFT);
            ADD_MOVE(INC_RIGHT);
        }
    }
    // NIWATORI
    {
        const auto size = pos.piece_list_size(turn, NIWATORI);
        REP(index, size) {
            const auto from = pos.piece_list(turn, NIWATORI, index);
            ADD_MOVE(INC_UP);
            ADD_MOVE(INC_DOWN);
            ADD_MOVE(INC_LEFT);
            ADD_MOVE(INC_RIGHT);
            if (turn == BLACK) {
                ADD_MOVE(INC_LEFTUP);
                ADD_MOVE(INC_RIGHTUP);
            } else {
                ADD_MOVE(INC_LEFTDOWN);
                ADD_MOVE(INC_RIGHTDOWN);
            }
        }
    }
#define ADD_MOVE_LION(dir) do {\
            const auto to = from + (dir);\
            if (attack::can_move(pos.square(to), move_flag) && !attack::is_attacked(pos, to, opp)) {\
                ml.add(move(from, to));\
            }\
}while(false)
    // LION
    {
        const auto size = pos.piece_list_size(turn, LION);
        REP(index, size) {
            const auto from = pos.piece_list(turn, LION, index);
            ADD_MOVE_LION(INC_UP);
            ADD_MOVE_LION(INC_DOWN);
            ADD_MOVE_LION(INC_LEFT);
            ADD_MOVE_LION(INC_RIGHT);
            ADD_MOVE_LION(INC_LEFTUP);
            ADD_MOVE_LION(INC_RIGHTUP);
            ADD_MOVE_LION(INC_LEFTDOWN);
            ADD_MOVE_LION(INC_RIGHTDOWN);
        }
    }
#undef ADD_MOVE
#undef ADD_MOVE_LION
}

}
#endif