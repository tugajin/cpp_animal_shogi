#ifndef __MOVEEVASION_HPP__
#define __MOVEEVASION_HPP__

#include "game.hpp"
#include "movelist.hpp"
#include "attack.hpp"
#include <unordered_map>

namespace gen {

template<bool is_exists = false> bool evasion_moves(const game::Position &pos, movelist::MoveList &ml) {
    const auto turn = pos.turn();
    const auto opp = change_turn(turn);
    const auto move_flag = (turn == BLACK) ? (BLACK_FLAG | COLOR_WALL_FLAG) : (WHITE_FLAG | COLOR_WALL_FLAG);
    const auto color_flag = (turn == BLACK) ? BLACK_FLAG : WHITE_FLAG;
    const attack::Checker checker(pos);
    ASSERT2(checker.num == 1,{
        Tee<<pos<<std::endl;
        Tee<<checker.num<<std::endl;
    });
    //王手駒を取る（らいおん以外で取る）
    {
        const auto to = checker.from[checker.num-1];
        const auto from = to + INC_DOWN;
        const auto color_piece = pos.square(from);
        const auto piece = to_piece(color_piece);
        const auto flag = (UP_FLAG | color_flag);
        if (piece != LION && (color_piece & flag) == flag) {
            if (color_piece == BLACK_HIYOKO && sq_rank(to) == RANK_1) {
                ASSERT(turn == BLACK);
                if (is_exists) { return true; }
                ml.add(move(from, to, true));
            } 
            if (is_exists) { return true; }
            ml.add(move(from, to));
        }
    }
    {
        const auto to = checker.from[checker.num-1];
        const auto from = to + INC_UP;
        const auto color_piece = pos.square(from);
        const auto piece = to_piece(color_piece);
        const auto flag = (DOWN_FLAG | color_flag);
        if (piece != LION && (color_piece & flag) == flag) {
            if (color_piece == WHITE_HIYOKO && sq_rank(to) == RANK_4) {
                ASSERT(turn == WHITE);
                if (is_exists) { return true; }
                ml.add(move(from, to, true));
            }
            if (is_exists) { return true; }
            ml.add(move(from, to));
        }
    }
#define FIND_CAPTURE(d, flag) do{\
        const auto to = checker.from[checker.num-1];\
        const auto from = to + (d);\
        const auto color_piece = pos.square(from);\
        const auto piece = to_piece(color_piece);\
        if (piece != LION && (color_piece & (flag)) == (flag)) {\
            if (is_exists) { return true; }\
            ml.add(move(from, to));\
        }\
} while(false)
    FIND_CAPTURE(INC_LEFT, RIGHT_FLAG | color_flag);
    FIND_CAPTURE(INC_RIGHT, LEFT_FLAG | color_flag);
    FIND_CAPTURE(INC_LEFTUP, RIGHTDOWN_FLAG | color_flag);
    FIND_CAPTURE(INC_LEFTDOWN, RIGHTUP_FLAG | color_flag);
    FIND_CAPTURE(INC_RIGHTUP, LEFTDOWN_FLAG | color_flag);
    FIND_CAPTURE(INC_RIGHTDOWN, LEFTUP_FLAG | color_flag);

#undef FIND_CAPTURE

#define ADD_MOVE_LION(dir) do {\
            const auto to = from + (dir);\
            if (attack::can_move(pos.square(to), move_flag) && !attack::is_attacked(pos, to, opp)) {\
                if (is_exists) { return true; }\
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
#undef ADD_MOVE_LION
    return false;
}

}
#endif