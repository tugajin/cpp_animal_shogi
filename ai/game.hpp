#ifndef __GAME_HPP__
#define __GAME_HPP__

#include <array>
#include <bitset>
#include <functional>
#include "common.hpp"
#include "util.hpp"
#include "movelist.hpp"
namespace game {
class Position;
}
namespace hash {
Key hash_key(const game::Position &pos);
}

namespace game {

class Position {
public:
    Position() {}
    Position(const ColorPiece pieces[], const Hand hand[], const Color turn);
    void init_pos();
    bool is_lose() const ;
    bool is_win() const;
    bool is_draw() const {
        if (this->ply_ >= 400) {
            return true;
        }
        const auto curr_key = this->history_[this->ply_];
        for (auto ply = this->ply_ - 4; ply >= 0; ply -= 2) {
            const auto prev_key = this->history_[ply];
            if (curr_key == prev_key) {
                return true;
            }
        }
        return false;
    }
    bool is_draw_short() const {
        if (this->ply_ >= 400) {
            return true;
        }
        return false;
    }
    bool is_done() const {
        return (this->is_lose() || this->is_draw());
    }
    Color turn() const {
        return this->turn_;
    }
    Square piece_list(const Color c, const Piece p, const int index) const {
        ASSERT(color_is_ok(c));
        ASSERT(piece_is_ok(p));
        ASSERT(index >= 0 || index < 2);
        return this->piece_list_[c][p][index];
    }
    int piece_list_size(const Color c, const Piece p) const {
        ASSERT(color_is_ok(c));
        ASSERT(piece_is_ok(p));
        return this->piece_list_size_[c][p];
    }
    ColorPiece square(const Square sq) const {
        ASSERT(sq >= SQ_BEGIN);
        ASSERT(sq < SQ_END);
        return this->square_[sq];
    }
    Hand hand(const Color c) const {
        ASSERT(color_is_ok(c));
        return this->hand_[c];
    }
    bool is_ok() const;
    std::string str() const;
    Square king_sq (const Color c) const {
        return this->piece_list_[c][LION][0];
    }
	friend std::ostream& operator<<(std::ostream& os, const Position& pos) {
        os << pos.str();
		return os;
	}
    Position next(const Move action) const;
    void dump() const;
    Key history() const {
        return this->history_[this->ply_];
    }
private:
    ColorPiece square_[SQ_END];//どんな駒が存在しているか？
    int piece_list_index_[SQ_END];//piece_listの何番目か

    Square piece_list_[COLOR_SIZE][PIECE_END][2];//駒がどのsqに存在しているか?
    int piece_list_size_[COLOR_SIZE][PIECE_END];//piece_listのどこまで使ったか？
    Hand hand_[COLOR_SIZE];

    Key history_[1024];
    int ply_;
    Color turn_;
    void quiet_move_piece(const Square from, const Square to, ColorPiece color_piece);
    void put_piece(const Square sq, const ColorPiece color_piece);
    void remove_piece(const Square sq, const ColorPiece color_piece);
};

Position::Position(const ColorPiece pieces[], const Hand hand[], const Color turn) {
   REP(i, SQ_END) {
        this->square_[i] = COLOR_WALL;
        this->piece_list_index_[i] = -1;
    }
    REP_COLOR(col) {
        REP_PIECE(pc) {
            this->piece_list_[col][pc][0] = this->piece_list_[col][pc][1] = SQ_WALL;
            this->piece_list_size_[col][pc] = 0;
        }
    }
    REP(i, 1024) {
        this->history_[i] = 0;
    }
    this->ply_ = 0;
    REP(i, SQUARE_SIZE) {
        if (pieces[i] != COLOR_EMPTY) {
            this->put_piece(SQUARE_INDEX[i], pieces[i]);
        } else {
            this->square_[SQUARE_INDEX[i]] = COLOR_EMPTY;
        }
    }
    this->hand_[BLACK] = hand[BLACK];
    this->hand_[WHITE] = hand[WHITE];
    this->turn_ = turn;
    this->history_[this->ply_] = hash::hash_key(*this);
}

void Position::quiet_move_piece(const Square from, const Square to, const ColorPiece color_piece) {
    ASSERT2(sq_is_ok(from),{ Tee<<"sq1:"<<from<<std::endl; });
    ASSERT2(sq_is_ok(to),{ Tee<<"sq2:"<<to<<std::endl; });
    ASSERT(color_piece != COLOR_EMPTY);
    const auto col = piece_color(color_piece);
    const auto pc = to_piece(color_piece);
    this->square_[from] = COLOR_EMPTY;
    this->square_[to] = color_piece;
    this->piece_list_index_[to] = this->piece_list_index_[from];
    this->piece_list_[col][pc][this->piece_list_index_[from]] = to;
    this->piece_list_index_[from] = -1;
}

void Position::put_piece(const Square sq, const ColorPiece color_piece) {
    ASSERT2(sq_is_ok(sq),{ Tee<<"sq:"<<sq<<std::endl; });
    ASSERT(color_piece != COLOR_EMPTY);
    const auto col = piece_color(color_piece);
    const auto pc = to_piece(color_piece);
    this->square_[sq] = color_piece;
    this->piece_list_index_[sq] = this->piece_list_size_[col][pc];
    this->piece_list_[col][pc][this->piece_list_size_[col][pc]] = sq;
    ++this->piece_list_size_[col][pc];
}

void Position::remove_piece(const Square sq, const ColorPiece color_piece) {
    const auto col = piece_color(color_piece);
    const auto pc = to_piece(color_piece);
    --this->piece_list_size_[col][pc];
    
    const auto last_sq = this->piece_list_[col][pc][this->piece_list_size_[col][pc]];
    const auto removed_index = this->piece_list_index_[sq];
    this->piece_list_[col][pc][removed_index] = last_sq;
    this->piece_list_index_[last_sq] = removed_index;

    this->square_[sq] = COLOR_EMPTY;
    this->piece_list_index_[sq] = -1;
}

std::string Position::str() const {
    std::string ret = "手番:" + color_str(this->turn()) + " hash:" + to_string(this->history_[this->ply_]) + " ply:" + to_string(this->ply_) + "\n";
    ret += "後手:" + hand_str(this->hand_[WHITE]) + "\n";
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto color_piece = this->square_[*p];
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
    ret += "先手:" + hand_str(this->hand_[BLACK]) ;
    return ret;
}

bool Position::is_ok() const {
    if (this->turn() != BLACK && this->turn() != WHITE) {
#if DEBUG
                Tee<<"error turn\n";
#endif
        return false;
    }
    if (this->ply_ < 0 || this->ply_ > 1024) {
#if DEBUG
                Tee<<"error history\n";
#endif
        return false;
    }
    // square -> piece_list
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        const auto sq = *p;
        const auto color_piece =this->square_[sq];
        if (color_piece != COLOR_EMPTY) {
            const auto piece = to_piece(color_piece);
            const auto color = piece_color(color_piece);
            const auto index = this->piece_list_index_[sq];
            if (index < 0 || index > 1) {
#if DEBUG
                Tee<<"error1\n";
#endif
                return false;
            }
            if (this->piece_list_[color][piece][index] == SQ_WALL) {
#if DEBUG
                Tee<<"error1.5\n";
                Tee<<this->piece_list_[color][piece][index]<<std::endl;
                Tee<<"color:"<<color<<std::endl;
                Tee<<"piece:"<<piece<<std::endl;
                Tee<<"index:"<<index<<std::endl;
                Tee<<"sq:"<<sq<<std::endl;
                return false;
#endif              
            }
            if (this->piece_list_[color][piece][index] != sq) {
#if DEBUG
                Tee<<"error2\n";
                Tee<<"sq:"<<sq<<std::endl;
                Tee<<this->piece_list_[color][piece][index]<<std::endl;
                Tee<<index<<std::endl;
                Tee<<color<<std::endl;
                Tee<<piece<<std::endl;
#endif
                return false;
            }
        } else {
            const auto index = this->piece_list_index_[sq];
            if (index != -1) {
#if DEBUG
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
            if (this->piece_list_size_[col][piece] != 0
            && this->piece_list_size_[col][piece] != 1
            && this->piece_list_size_[col][piece] != 2) {
#if DEBUG
                Tee<<"error size\n";
                Tee<<this->piece_list_size_[col][piece]<<std::endl;
                Tee<<col<<std::endl;
                Tee<<piece<<std::endl;
#endif              
                return false;
            }
            REP(index, this->piece_list_size_[col][piece]) {
                const auto piece_list_sq = this->piece_list_[col][piece][index];
                if (color_piece(piece,col) != this->square_[piece_list_sq]) {
#if DEBUG
                Tee<<"error4\n";
#endif
                    return false;
                }
                if (this->piece_list_index_[piece_list_sq] != index) {
#if DEBUG
                Tee<<"error5\n";
#endif
                    return false;
                }
            }
        }
    }
    if (!hand_is_ok(this->hand_[BLACK])) {
#if DEBUG
                Tee<<"error black hand\n";
#endif
        return false;
    } 
    if (!hand_is_ok(this->hand_[WHITE])) {
#if DEBUG
                Tee<<"error white hand\n";
#endif
        return false;
    } 

    return true;
}

Position Position::next(const Move action) const {
    ASSERT2(this->is_ok(),{
        Tee<<"prev_next\n";
        Tee<<this->str()<<std::endl;
        Tee<<move_str(action)<<std::endl;
    });
    Position next_pos = *this;
    const auto turn = next_pos.turn();
    if (move_is_drop(action)) {
        const auto piece = move_piece(action);
        const auto to = move_to(action);
        next_pos.put_piece(to, color_piece(piece,turn));
        next_pos.hand_[turn] = dec_hand(next_pos.hand_[turn], piece);
    } else {
        const auto from = move_from(action);
        const auto to = move_to(action);
        const auto src_color_piece = next_pos.square(from);
        auto dst_color_piece = src_color_piece;
        const auto captured_color_piece = next_pos.square(to);
        if (captured_color_piece != COLOR_EMPTY) {
            next_pos.remove_piece(to, captured_color_piece);
            next_pos.hand_[turn] = inc_hand(next_pos.hand_[turn], unprom(to_piece(captured_color_piece)));
        }
        if (move_is_prom(action)) {
            const auto src_piece = to_piece(src_color_piece);
            const auto dst_piece = prom(src_piece);
            dst_color_piece = color_piece(dst_piece, turn);
            next_pos.remove_piece(from, src_color_piece);
            next_pos.put_piece(to, dst_color_piece);
        } else {
            next_pos.quiet_move_piece(from, to, src_color_piece);
        }
    }
    ++next_pos.ply_;
    next_pos.turn_ = change_turn(turn);
    next_pos.history_[next_pos.ply_] = hash::hash_key(next_pos);
    ASSERT2(next_pos.is_ok(),{
        //Tee<<"prev\n";
        //Tee<<*this<<std::endl;
        Tee<<"after next\n";
        Tee<<next_pos<<std::endl;
        Tee<<move_str(action)<<std::endl;
        this->dump();
    });
    return next_pos;
}

void Position::dump() const {
    Tee<<"dump:\n";
    Tee<<"square\n";
    REP(sq, SQ_END) {
        if (sq % 8 == 0) { Tee<<"\n";}
        Tee<<to_string(static_cast<int>(this->square_[sq]))+",";
    }
    Tee<<"\nsquare_index\n";
    REP(sq, SQ_END) {
        if (sq % 8 == 0) { Tee<<"\n";}
        Tee<<to_string(static_cast<int>(this->piece_list_index_[sq]))+",";
    }
    Tee<<"\npiece_size\n";
    REP_COLOR(col) {
        REP_PIECE(pc) {
            Tee << to_string(this->piece_list_size_[col][pc])+",";
        }
        Tee<<"\n";
    }
    Tee<<"piece_list\n";
    REP_COLOR(col) {
        REP_PIECE(pc) {
            Tee<<piece_str(Piece(pc))<<":";
            REP(index, 2) {
                Tee<<to_string(this->piece_list_[col][pc][index]) + ",";
            }
            Tee<<"\n";
        }
    }
    Tee<<"black_hand:"<<static_cast<int>(this->hand_[BLACK])<<std::endl;
    Tee<<"white_hand:"<<static_cast<int>(this->hand_[WHITE])<<std::endl;
}

void test_common() {
#if DEBUG
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
#endif
}
}
#endif