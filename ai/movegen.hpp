#ifndef __MOVEGEN__HPP__
#define __MOVEGEN_HPP__

#include "game.hpp"
#include "movelist.hpp"
namespace gen {
bool can_move(const ColorPiece p, const int move_flag) {
    return (p & move_flag) == 0;
}
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
void pos_moves(const game::Position &pos, movelist::MoveList &ml) {
    const auto turn = pos.turn();
    const auto move_flag = (turn == BLACK) ? (BLACK_FLAG | COLOR_WALL_FLAG) : (WHITE_FLAG | COLOR_WALL_FLAG);
    // HIYOKO
    {
        const auto size = pos.piece_list_size(turn, HIYOKO);
        REP(index, size) {
            const auto from = pos.piece_list(turn, HIYOKO, index);
            if (turn == BLACK) {
                const auto to = from + SQ_UP;
                if (can_move(pos.square(to), move_flag)) {
                    const auto prom = sq_rank(to) == RANK_1;
                    if (prom) {
                        ml.add(move(from, to, prom));
                    }
                    ml.add(move(from, to));
                }
            } else {
                const auto to = from + SQ_DOWN;
                if (can_move(pos.square(to), move_flag)) {
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
            if (can_move(pos.square(to), move_flag)) {\
                ml.add(move(from, to));\
            }\
}while(false)
    // ZOU
    {
        const auto size = pos.piece_list_size(turn, ZOU);
        REP(index, size) {
            const auto from = pos.piece_list(turn, ZOU, index);
            ADD_MOVE(SQ_LEFTUP);
            ADD_MOVE(SQ_RIGHTUP);
            ADD_MOVE(SQ_LEFTDOWN);
            ADD_MOVE(SQ_RIGHTDOWN);
        }
    }
    // KIRIN
    {
        const auto size = pos.piece_list_size(turn, KIRIN);
        REP(index, size) {
            const auto from = pos.piece_list(turn, KIRIN, index);
            ADD_MOVE(SQ_UP);
            ADD_MOVE(SQ_DOWN);
            ADD_MOVE(SQ_LEFT);
            ADD_MOVE(SQ_RIGHT);
        }
    }
    // LION
    {
        const auto size = pos.piece_list_size(turn, LION);
        REP(index, size) {
            const auto from = pos.piece_list(turn, LION, index);
            ADD_MOVE(SQ_UP);
            ADD_MOVE(SQ_DOWN);
            ADD_MOVE(SQ_LEFT);
            ADD_MOVE(SQ_RIGHT);
            ADD_MOVE(SQ_LEFTUP);
            ADD_MOVE(SQ_RIGHTUP);
            ADD_MOVE(SQ_LEFTDOWN);
            ADD_MOVE(SQ_RIGHTDOWN);
        }
    }
#undef ADD_MOVE
}
void legal_moves(const game::Position &pos, movelist::MoveList &ml) {
    pos_moves(pos,ml);
    drop_moves(pos, ml);
}

void test_gen() {
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN, WHITE_LION, WHITE_ZOU,
            COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_HIYOKO, COLOR_EMPTY,
            BLACK_ZOU, BLACK_LION, BLACK_KIRIN,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        legal_moves(pos, ml);
        Tee<<ml<<std::endl;
    }
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN, WHITE_LION, WHITE_ZOU,
            COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_LION, COLOR_EMPTY,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        hand[BLACK] = inc_hand(hand[BLACK],HIYOKO);
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
        hand[BLACK] = inc_hand(hand[BLACK],KIRIN);
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        legal_moves(pos, ml);
        Tee<<ml<<std::endl;
    }
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN,  WHITE_LION, WHITE_ZOU,
            BLACK_HIYOKO, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY,  COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY,  BLACK_LION, COLOR_EMPTY,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        hand[BLACK] = inc_hand(hand[BLACK],HIYOKO);
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
        hand[BLACK] = inc_hand(hand[BLACK],KIRIN);
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        legal_moves(pos, ml);
        Tee<<ml<<std::endl;
    }

}

}
#endif