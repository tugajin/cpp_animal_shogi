#ifndef __GAME_HPP__
#define __GAME_HPP__

#include <array>
#include <bitset>
#include <functional>
#include "common.hpp"
#include "util.hpp"
#include "movelist.hpp"

namespace game {

class Position {
public:
    Position() {}
    Position(const ColorPiece pieces[], const Hand hand[], const Color turn);
    Position(const Key h);
    void init_pos();
    bool is_lose() const {
        return false;
    }
    bool is_win() const {
        return -1;
    }
    bool is_draw() const {
        return false;
    }
    bool is_done() const {
        return false;
    }
    Color turn() const {
        return this->pos_turn;
    }
    bool is_ok() const;
    Key hash_key() const;
    std::string str() const;
	friend std::ostream& operator<<(std::ostream& os, const Position& pos) {
        os << pos.str();
		return os;
	}
    Feature feature() const {
        Feature feat(FEAT_SIZE, std::vector<int>(FILE_SIZE, 0));
        return feat;
    }
    Position next(const Move action) const;
    void legal_moves(movelist::MoveList &ml) const;
private:
    ColorPiece square[SQ_END];//どんな駒が存在しているか？
    int piece_list_index[SQ_END];//piece_listの何番目か

    Square piece_list[COLOR_SIZE][PIECE_END][2];//駒がどのsqに存在しているか?
    int piece_list_size[COLOR_SIZE][PIECE_END];//piece_listのどこまで使ったか？
    Hand hand[COLOR_SIZE];

    Key history[1024];
    int history_sp;
    Color pos_turn;
    void put_piece(const Square sq, const ColorPiece color_piece);
    void remove_piece(const Square sq, const ColorPiece color_piece);
};

Position::Position(const ColorPiece pieces[], const Hand hand[], const Color turn) {
   REP(i, SQ_END) {
        this->square[i] = COLOR_EMPTY;
        this->piece_list_index[i] = -1;
    }
    REP_COLOR(col) {
        REP_PIECE(pc) {
            this->piece_list[col][pc][0] = this->piece_list[col][pc][1] = SQ_WALL;
            this->piece_list_size[col][pc] = 0;
        }
    }
    REP(i, 1024) {
        this->history[i] = 0;
    }
    this->history_sp = 0;
    REP(i, SQUARE_SIZE) {
        if (pieces[i] != COLOR_EMPTY) {
            this->put_piece(SQUARE_INDEX[i], pieces[i]);
        }
    }
    this->hand[BLACK] = hand[BLACK];
    this->hand[WHITE] = hand[WHITE];
    this->pos_turn = turn;
}

void Position::put_piece(const Square sq, const ColorPiece color_piece) {
    ASSERT2(sq_is_ok(sq),{ Tee<<"sq:"<<sq<<std::endl; });
    ASSERT(color_piece != COLOR_EMPTY);
    const auto col = piece_color(color_piece);
    const auto pc = to_piece(color_piece);
    this->square[sq] = color_piece;
    this->piece_list_index[sq] = this->piece_list_size[col][pc];
    this->piece_list[col][pc][this->piece_list_size[col][pc]] = sq;
    ++this->piece_list_size[col][pc];
}

void Position::remove_piece(const Square sq, const ColorPiece color_piece) {
    const auto col = piece_color(color_piece);
    const auto pc = to_piece(color_piece);
    this->square[sq] = COLOR_EMPTY;
    this->piece_list_index[sq] = -1;
    this->piece_list[col][pc][this->piece_list_size[col][pc]] = SQ_WALL;
    --this->piece_list_size[col][pc];
}

Position::Position(const Key key) {
    ColorPiece pieces[SQUARE_SIZE] = {};
    Hand hand[COLOR_SIZE];
    Color turn = BLACK;
    auto k = key;
    if (k & 0x1) {
        turn = WHITE;
    }
    auto num = (k >> 1) & 0x3;
    k >>= 1;
    REP(i, num) {
        hand[WHITE] = inc_hand(hand[WHITE],ZOU);
    }
    num = (k >> 2) & 0x3;
    k >>= 2;
    REP(i, num) {
        hand[WHITE] = inc_hand(hand[WHITE],KIRIN);
    }
    num = (k >>2 ) & 0x3;
    k >>= 2;
    REP(i, num) {
        hand[WHITE] = inc_hand(hand[WHITE],HIYOKO);
    }
    num = (k >>2 ) & 0x3;
    k >>= 2;
    REP(i, num) {
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
    }
    num = (k >> 2) & 0x3;
    k >>= 2;
    REP(i, num) {
        hand[BLACK] = inc_hand(hand[BLACK],KIRIN);
    }
    num = (k >>2 ) & 0x3;
    k >>= 2;
    REP(i, num) {
        hand[BLACK] = inc_hand(hand[BLACK],HIYOKO);
    }
    auto p = SQUARE_INDEX + SQUARE_SIZE;
    auto i = SQUARE_SIZE;
    for (auto p = SQUARE_INDEX + SQUARE_SIZE; p != SQUARE_INDEX; --p) {
        const auto color_piece = static_cast<ColorPiece>((k >> 4) & 0xF);
        k >>= 4;
        pieces[--i] = color_piece;
    }
    Position(pieces, hand, turn);
}

Key Position::hash_key() const {
    Key key = this->turn() == BLACK ? 0ull : 2ull;
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto color_piece = this->square[*p];
        key += (key << 4) + color_piece_no(color_piece);
    }
    key += (key << 2) + num_piece(this->hand[BLACK], HIYOKO);
    key += (key << 2) + num_piece(this->hand[BLACK], KIRIN);
    key += (key << 2) + num_piece(this->hand[BLACK], ZOU);
    
    key += (key << 2) + num_piece(this->hand[WHITE], HIYOKO);
    key += (key << 2) + num_piece(this->hand[WHITE], KIRIN);
    key += (key << 2) + num_piece(this->hand[WHITE], ZOU);

    return key;
}

std::string Position::str() const {
    std::string ret = "手番:" + color_str(this->turn()) + " hash:" + to_string(this->hash_key()) +"\n";
    ret += "後手:" + hand_str(this->hand[WHITE]) + "\n";
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto color_piece = this->square[*p];
        const auto col = piece_color(color_piece);
        const auto piece = to_piece(color_piece);
        if (piece == EMPTY) {
            ret += " ";
        } else if (col == BLACK) {
            ret += "^";
        } else {
            ret += "v";
        }
        ret += piece_str(piece);
        if (sq_file(*p) == FILE_1) { ret += "\n"; }
    }
    ret += "先手:" + hand_str(this->hand[BLACK]) + "\n";
    return ret;
}

bool Position::is_ok() const {
    if (this->turn() != BLACK && this->turn() != WHITE) {
#ifdef DEBUG
                Tee<<"error turn\n";
#endif
        return false;
    }
    if (this->history_sp < 0 || this->history_sp > 1024) {
#ifdef DEBUG
                Tee<<"error history\n";
#endif
        return false;
    }
    // square -> piece_list
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto sq = *p;
        const auto color_piece =this->square[sq];
        if (color_piece != COLOR_EMPTY) {
            const auto piece = to_piece(color_piece);
            const auto color = piece_color(color_piece);
            const auto index = this->piece_list_index[sq];
            if (index < 0 || index > 1) {
#ifdef DEBUG
                Tee<<"error1\n";
#endif
                return false;
            }
            if (this->piece_list[color][piece][index] != sq) {
#ifdef DEBUG
                Tee<<"error2\n";
#endif
                return false;
            }
        } else {
            const auto index = this->piece_list_index[sq];
            if (index != -1) {
#ifdef DEBUG
                Tee<<"error3\n";
                Tee<<sq<<std::endl;
                Tee<<index<<std::endl;
#endif
                return false;
            }
        }
    }
    // piece_list -> square
    REP_COLOR(col) {
        REP_PIECE(piece) {
            REP(index, this->piece_list_size[col][piece]) {
                const auto piece_list_sq = this->piece_list[col][piece][index];
                if (color_piece(piece,col) != this->square[piece_list_sq]) {
#ifdef DEBUG
                Tee<<"error4\n";
#endif
                    return false;
                }
                if (this->piece_list_index[piece_list_sq] != index) {
#ifdef DEBUG
                Tee<<"error5\n";
#endif
                    return false;
                }
            }
        }
    }
    if (!hand_is_ok(this->hand[BLACK])) {
#ifdef DEBUG
                Tee<<"error black hand\n";
#endif
        return false;
    } 
    if (!hand_is_ok(this->hand[WHITE])) {
#ifdef DEBUG
                Tee<<"error white hand\n";
#endif
        return false;
    } 
    return true;
}

Position Position::next(const Move action) const {
    return Position();
}
void Position::legal_moves(movelist::MoveList &ml) const {
}

Position from_hash(const Key h) {
    return Position(h);
}

void test_pos() {
    ColorPiece pieces[SQUARE_SIZE] = {
        WHITE_KIRIN, WHITE_LION, WHITE_ZOU,
        COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
        COLOR_EMPTY, BLACK_HIYOKO, COLOR_EMPTY,
        BLACK_ZOU, BLACK_LION, BLACK_KIRIN,
    };
    Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
    Position pos(pieces,hand,BLACK);
    ASSERT(pos.is_ok());
    Tee<<pos.str()<<std::endl;
}
void test_common() {
    ASSERT(change_turn(BLACK) == WHITE);
    ASSERT(change_turn(WHITE) == BLACK);

    ASSERT(square(FILE_1, RANK_1) == SQ_11);
    ASSERT(square(FILE_1, RANK_2) == SQ_12);
    ASSERT(square(FILE_1, RANK_3) == SQ_13);
    ASSERT(square(FILE_1, RANK_4) == SQ_14);
    
    ASSERT(square(FILE_2, RANK_1) == SQ_21);
    ASSERT(square(FILE_2, RANK_2) == SQ_22);
    ASSERT(square(FILE_2, RANK_3) == SQ_23);
    ASSERT(square(FILE_2, RANK_4) == SQ_24);

    ASSERT(square(FILE_3, RANK_1) == SQ_31);
    ASSERT(square(FILE_3, RANK_2) == SQ_32);
    ASSERT(square(FILE_3, RANK_3) == SQ_33);
    ASSERT(square(FILE_3, RANK_4) == SQ_34);

    ASSERT(sq_file(SQ_11) == FILE_1);    
    ASSERT(sq_file(SQ_12) == FILE_1);    
    ASSERT(sq_file(SQ_13) == FILE_1);    
    ASSERT(sq_file(SQ_14) == FILE_1);

    ASSERT(sq_file(SQ_21) == FILE_2);    
    ASSERT(sq_file(SQ_22) == FILE_2);    
    ASSERT(sq_file(SQ_23) == FILE_2);    
    ASSERT(sq_file(SQ_24) == FILE_2);    

    ASSERT(sq_file(SQ_31) == FILE_3);    
    ASSERT(sq_file(SQ_32) == FILE_3);    
    ASSERT(sq_file(SQ_33) == FILE_3);    
    ASSERT(sq_file(SQ_34) == FILE_3);
    
    ASSERT(sq_rank(SQ_11) == RANK_1);    
    ASSERT(sq_rank(SQ_12) == RANK_2);    
    ASSERT(sq_rank(SQ_13) == RANK_3);    
    ASSERT(sq_rank(SQ_14) == RANK_4);

    ASSERT(sq_rank(SQ_21) == RANK_1);    
    ASSERT(sq_rank(SQ_22) == RANK_2);    
    ASSERT(sq_rank(SQ_23) == RANK_3);    
    ASSERT(sq_rank(SQ_24) == RANK_4);    

    ASSERT(sq_rank(SQ_31) == RANK_1);    
    ASSERT(sq_rank(SQ_32) == RANK_2);    
    ASSERT(sq_rank(SQ_33) == RANK_3);    
    ASSERT(sq_rank(SQ_34) == RANK_4);
    
    auto h = HAND_NONE;
    ASSERT(!has_piece(h,HIYOKO));
    ASSERT(!has_piece(h,KIRIN));
    ASSERT(!has_piece(h,ZOU));

    auto h_hiyoko = inc_hand(h,HIYOKO);
    ASSERT(has_piece(h_hiyoko,HIYOKO));
    ASSERT(!has_piece(h_hiyoko,KIRIN));
    ASSERT(!has_piece(h_hiyoko,ZOU));

    auto h_hiyoko2 = inc_hand(h_hiyoko, HIYOKO);
    ASSERT(has_piece(h_hiyoko2,HIYOKO));
    ASSERT(!has_piece(h_hiyoko2,KIRIN));
    ASSERT(!has_piece(h_hiyoko2,ZOU));

    ASSERT(dec_hand(h_hiyoko2, HIYOKO) == h_hiyoko);
    ASSERT(dec_hand(h_hiyoko, HIYOKO) == h);

    auto h_kirin = inc_hand(h, KIRIN);
    ASSERT(!has_piece(h_kirin,HIYOKO));
    ASSERT(has_piece(h_kirin,KIRIN));
    ASSERT(!has_piece(h_kirin,ZOU));

    auto h_kirin2 = inc_hand(h_kirin, KIRIN);
    ASSERT(!has_piece(h_kirin2,HIYOKO));
    ASSERT(has_piece(h_kirin2,KIRIN));
    ASSERT(!has_piece(h_kirin2,ZOU));

    ASSERT(dec_hand(h_kirin2, KIRIN) == h_kirin);
    ASSERT(dec_hand(h_kirin, KIRIN) == h);

    auto h_zou = inc_hand(h, ZOU);
    ASSERT(!has_piece(h_zou,HIYOKO));
    ASSERT(!has_piece(h_zou,KIRIN));
    ASSERT(has_piece(h_zou,ZOU));

    auto h_zou2 = inc_hand(h_zou, ZOU);
    ASSERT(!has_piece(h_zou2,HIYOKO));
    ASSERT(!has_piece(h_zou2,KIRIN));
    ASSERT(has_piece(h_zou2,ZOU));

    ASSERT(dec_hand(h_zou2, ZOU) == h_zou);
    ASSERT(dec_hand(h_zou, ZOU) == h);

    auto h_all = inc_hand(h, HIYOKO);
    h_all = inc_hand(h_all, KIRIN);
    h_all = inc_hand(h_all, ZOU);
    
    ASSERT(has_piece(h_all,HIYOKO));
    ASSERT(has_piece(h_all,KIRIN));
    ASSERT(has_piece(h_all,ZOU));
    
    h_all = dec_hand(h_all, HIYOKO);
    h_all = dec_hand(h_all, KIRIN);
    h_all = dec_hand(h_all, ZOU);

    ASSERT(h_all == h);
    for(auto *fp = SQUARE_INDEX; *fp != SQ_WALL; ++fp) {
        // 持ち駒を打つ手
        const auto from = *fp;

        const auto mh = move(from, HIYOKO);
        const auto mk = move(from, KIRIN);
        const auto mz = move(from, ZOU);

        ASSERT(move_to(mh) == from);
        ASSERT(move_to(mk) == from);
        ASSERT(move_to(mz) == from);

        ASSERT(move_is_drop(mh));
        ASSERT(move_is_drop(mk));
        ASSERT(move_is_drop(mz));

        ASSERT(!move_is_prom(mh));
        ASSERT(!move_is_prom(mk));
        ASSERT(!move_is_prom(mz));

        ASSERT(move_is_ok(mh));
        ASSERT(move_is_ok(mk));
        ASSERT(move_is_ok(mz));

        ASSERT(sq_is_ok(from));
        
        // 盤上の駒を動かす手
        for(auto *tp = SQUARE_INDEX; *tp != SQ_WALL; ++tp) {
            const auto to = *tp;
            const auto m = move(from, to);
            const auto m2 = move(from, to, true);
            const auto from2 = move_from(m);
            const auto to2 = move_to(m);
            ASSERT(from == from2);
            ASSERT(to == to2);
            ASSERT(!move_is_prom(m));
            ASSERT(!move_is_drop(m));
            ASSERT(move_is_prom(m2));
            ASSERT(!move_is_drop(m2));
            ASSERT(move_is_ok(m));
            ASSERT(move_is_ok(m2));
        }
    }
    ASSERT(prom(HIYOKO) == NIWATORI);
    ASSERT(unprom(NIWATORI) == HIYOKO);
    ASSERT(unprom(ZOU) == ZOU);
    ASSERT(unprom(KIRIN) == KIRIN);
    ASSERT(unprom(LION) == LION);

    ASSERT(color_piece(HIYOKO,BLACK) == BLACK_HIYOKO);
    ASSERT(color_piece(KIRIN,BLACK) == BLACK_KIRIN);
    ASSERT(color_piece(ZOU, BLACK) == BLACK_ZOU);
    ASSERT(color_piece(LION, BLACK) == BLACK_LION);
    ASSERT(color_piece(NIWATORI, BLACK) == BLACK_NIWATORI);

    ASSERT(color_piece(HIYOKO, WHITE) == WHITE_HIYOKO);
    ASSERT(color_piece(KIRIN, WHITE) == WHITE_KIRIN);
    ASSERT(color_piece(ZOU, WHITE) == WHITE_ZOU);
    ASSERT(color_piece(LION, WHITE) == WHITE_LION);
    ASSERT(color_piece(NIWATORI, WHITE) == WHITE_NIWATORI);

}
void test_nn() {
}
}
#endif