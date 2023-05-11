#ifndef __ATTACK_HPP__
#define __ATTACK_HPP__

#include "game.hpp"

namespace attack {

class Attacker {
public:
    int num;//どうぶつしょうぎの場合は絶対1
    Square from[1];
    Square dir[1];
    Attacker(const game::Position &pos) {
        this->num = 0;
        const auto turn = pos.turn();
        const auto opp = change_turn(turn);
        const auto color_flag = (opp == BLACK) ? BLACK_FLAG : WHITE_FLAG;
        const auto sq = pos.king_sq(turn);

#define SET_ATTACK(d, flag) do{\
        const auto from = sq + (d);\
        const auto color_piece = pos.square(from);\
        if ((color_piece & (flag)) == (flag)) {\
            this->from[this->num] = from;\
            this->dir[this->num] = (d);\
            ++this->num;\
            /*王手はひとつだけなので*/return;\
        }\
} while(false)

        SET_ATTACK(INC_DOWN, UP_FLAG | color_flag);
        SET_ATTACK(INC_UP, DOWN_FLAG | color_flag);
        SET_ATTACK(INC_LEFT, RIGHT_FLAG | color_flag);
        SET_ATTACK(INC_RIGHT, LEFT_FLAG | color_flag);
        SET_ATTACK(INC_LEFTUP, RIGHTDOWN_FLAG | color_flag);
        SET_ATTACK(INC_LEFTDOWN, RIGHTUP_FLAG | color_flag);
        SET_ATTACK(INC_RIGHTUP, LEFTDOWN_FLAG | color_flag);
        SET_ATTACK(INC_RIGHTDOWN, LEFTUP_FLAG | color_flag);

#undef SET_ATTACK
    }
};

inline bool is_attacked(const game::Position &pos, const Square sq, const Color turn) {
#define CHECK_ATTACK(dir, flag) do{\
        const auto from = sq + (dir);\
        const auto color_piece = pos.square(from);\
        if ((color_piece & (flag)) == (flag)) {\
            return true;\
        }\
} while(false)
    const auto color_flag = (turn == BLACK) ? BLACK_FLAG : WHITE_FLAG;
    CHECK_ATTACK(INC_DOWN, UP_FLAG | color_flag);
    CHECK_ATTACK(INC_UP, DOWN_FLAG | color_flag);
    CHECK_ATTACK(INC_LEFT, RIGHT_FLAG | color_flag);
    CHECK_ATTACK(INC_RIGHT, LEFT_FLAG | color_flag);
    CHECK_ATTACK(INC_LEFTUP, RIGHTDOWN_FLAG | color_flag);
    CHECK_ATTACK(INC_LEFTDOWN, RIGHTUP_FLAG | color_flag);
    CHECK_ATTACK(INC_RIGHTUP, LEFTDOWN_FLAG | color_flag);
    CHECK_ATTACK(INC_RIGHTDOWN, LEFTUP_FLAG | color_flag);

#undef CHECK_ATTACK

    return false;
}

inline bool in_checked(const game::Position &pos) {
    return is_attacked(pos,pos.king_sq(pos.turn()),change_turn(pos.turn()));
}

inline bool can_move(const ColorPiece p, const int move_flag) {
    return (p & move_flag) == 0;
}

inline void test_attack() {
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN, WHITE_LION, WHITE_ZOU,
            COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_HIYOKO, COLOR_EMPTY,
            BLACK_ZOU, BLACK_LION, BLACK_KIRIN,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        game::Position pos(pieces,hand,BLACK);
        ASSERT(!is_attacked(pos,SQ_11, BLACK));
        ASSERT(!is_attacked(pos,SQ_12, BLACK));
        ASSERT(is_attacked(pos,SQ_13, BLACK));
        ASSERT(is_attacked(pos,SQ_14, BLACK));

        ASSERT(!is_attacked(pos,SQ_21, BLACK));
        ASSERT(is_attacked(pos,SQ_22, BLACK));
        ASSERT(is_attacked(pos,SQ_23, BLACK));
        ASSERT(is_attacked(pos,SQ_24, BLACK));

        ASSERT(!is_attacked(pos,SQ_31, BLACK));
        ASSERT(!is_attacked(pos,SQ_32, BLACK));
        ASSERT(is_attacked(pos,SQ_33, BLACK));
        ASSERT(is_attacked(pos,SQ_34, BLACK));

        ASSERT(is_attacked(pos,SQ_11, WHITE));
        ASSERT(is_attacked(pos,SQ_12, WHITE));
        ASSERT(!is_attacked(pos,SQ_13, WHITE));
        ASSERT(!is_attacked(pos,SQ_14, WHITE));

        ASSERT(is_attacked(pos,SQ_21, WHITE));
        ASSERT(is_attacked(pos,SQ_22, WHITE));
        ASSERT(is_attacked(pos,SQ_23, WHITE));
        ASSERT(!is_attacked(pos,SQ_24, WHITE));

        ASSERT(is_attacked(pos,SQ_31, WHITE));
        ASSERT(is_attacked(pos,SQ_32, WHITE));
        ASSERT(!is_attacked(pos,SQ_33, WHITE));
        ASSERT(!is_attacked(pos,SQ_34, WHITE));
    }
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN, WHITE_LION, WHITE_ZOU,
            COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
            COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
            BLACK_ZOU, BLACK_LION, BLACK_KIRIN,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        game::Position pos(pieces,hand,BLACK);
        ASSERT(in_checked(pos));
    }

}
}
namespace game{
bool Position::is_win() const {
    //相手のライオンが取れそうなら勝ち
    const auto me = this->turn();
    const auto opp = change_turn(me);
    if (attack::is_attacked(*this, this->king_sq(opp), me)) {
        return true;
    }
    //自分のライオンがトライしていたら勝ち
    const auto lion_sq = this->king_sq(me);
    if ((me == BLACK && sq_rank(lion_sq) == RANK_1) || (me == WHITE && sq_rank(lion_sq) == RANK_4)){
        return !attack::is_attacked(*this, lion_sq, opp);
    } 
    return false;
}
bool Position::is_lose() const {
    //相手のライオンがトライしていたら負け
    const auto me = this->turn();
    const auto opp = change_turn(me);
    const auto lion_sq = this->king_sq(opp);
    if ((opp == BLACK && sq_rank(lion_sq) == RANK_1) || (opp == WHITE && sq_rank(lion_sq) == RANK_4)){
        if (!attack::is_attacked(*this, lion_sq, me)) {
            return true;
        }
    } 
    //盤上に自分のライオンがなかったら負け
    const auto my_lion = (this->turn() == BLACK) ? BLACK_LION : WHITE_LION;
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto color_piece = this->square(*p);
        if (color_piece == my_lion) {
            return false;
        }
    }
    return true;
}
}
#endif