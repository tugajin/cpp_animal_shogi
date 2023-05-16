#ifndef __HASH_HPP__
#define __HASH_HPP__
#include "game.hpp"

namespace hash {

inline constexpr Key START_HASH_KEY = 1248110771415179264UL;

game::Position from_hash(const Key key) {
    ColorPiece pieces[SQUARE_SIZE] = {};
    Hand hand[COLOR_SIZE] = {HAND_NONE, HAND_NONE};
    Color turn = BLACK;
    auto k = key;
    if (k & 0x1) {
        turn = WHITE;
    }
    k >>= 1;
    auto num = static_cast<int>(k & 0x3);
    REP(i, num) {
        hand[WHITE] = inc_hand(hand[WHITE],ZOU);
    }
    k >>= 2;
    num = static_cast<int>(k & 0x3);
    REP(i, num) {
        hand[WHITE] = inc_hand(hand[WHITE],KIRIN);
    }
    k >>= 2;
    num = static_cast<int>(k & 0x3);
    REP(i, num) {
        hand[WHITE] = inc_hand(hand[WHITE],HIYOKO);
    }
    k >>= 2;
    num = static_cast<int>(k & 0x3);
    REP(i, num) {
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
    }
    k >>= 2;
    num = static_cast<int>(k & 0x3);
    REP(i, num) {
        hand[BLACK] = inc_hand(hand[BLACK],KIRIN);
    }
    k >>= 2;
    num = static_cast<int>(k & 0x3);
    REP(i, num) {
        hand[BLACK] = inc_hand(hand[BLACK],HIYOKO);
    }
    k >>= 2;
    for (auto i = SQUARE_SIZE-1; i>=0; i--) {
        const auto color_piece = piece_no_color_piece(k & 0xF);
        k >>= 4;
        pieces[i] = color_piece;
    }
    return game::Position(pieces, hand, turn);
}
Key hash_key(const game::Position &pos) {
    Key key = pos.turn() == BLACK ? 0ull : 0ull;
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto color_piece = pos.square(*p);
        key += color_piece_no(color_piece);
        key <<= 4;
    }
    key >>= 4;
    key = (key << 2) + num_piece(pos.hand(BLACK), HIYOKO);
    key = (key << 2) + num_piece(pos.hand(BLACK), KIRIN);
    key = (key << 2) + num_piece(pos.hand(BLACK), ZOU);
    
    key = (key << 2) + num_piece(pos.hand(WHITE), HIYOKO);
    key = (key << 2) + num_piece(pos.hand(WHITE), KIRIN);
    key = (key << 2) + num_piece(pos.hand(WHITE), ZOU);
    
    key = (pos.turn() == BLACK) ? (key << 1) + 0 : (key << 1) + 1;

    return key;
}

game::Position hirate() {
    return from_hash(START_HASH_KEY);
}

void test_hash() {
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            CE,WL,CE,
            CE,WH,CE,
            CE,BH,CE,
            CE,BL,CE,
        };
        Hand hand[COLOR_SIZE] = { HAND_KIRIN, HAND_ZOU };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        auto new_pos = from_hash(hash::hash_key(pos));
        Tee<<new_pos<<std::endl;
    }
    {
        ColorPiece pieces[SQUARE_SIZE] = {
            WK,WL,WZ,
            CE,WH,CE,
            CE,BH,CE,
            BZ,BL,BK,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        auto new_pos = from_hash(hash::hash_key(pos));
        Tee<<new_pos<<std::endl;
    }
}
}
#endif