#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "util.hpp"

// 0	1	2	3	4	5	6	7
// 8	9	v10	v11	v12	13	14	15
// 16	17	v18	v19	v20	21	22	23
// 24	25	v26	v27	v28	29	30	31
// 32	33	v34	v35	v36	37	38	39
// 40	41	42	43	44	45	46	47


constexpr inline int FEAT_SIZE = 10;
constexpr inline int SQUARE_SIZE = 12;

#define REP(i,e) for (auto (i) = 0; (i) < (e); ++(i))
#define REP_FILE(i) for (auto (i) = FILE_3; (i) < FILE_SIZE; ++(i)) 
#define REP_RANK(i) for (auto (i) = RANK_1; (i) < RANK_SIZE; ++(i)) 
#define REP_COLOR(i) for (auto (i) = BLACK; (i) < COLOR_SIZE; ++(i)) 
#define REP_PIECE(i) for (auto (i) = HIYOKO; (i) < PIECE_END; ++(i)) 

typedef std::vector<std::vector<int>> Feature;
typedef double NNScore;
typedef uint64 Key;

enum Move : int {
    MOVE_NONE = -1
};

enum Color : int {
    BLACK = 0, WHITE = 1, COLOR_SIZE = 2,
};

enum File : int {
    FILE_1 = 2, 
    FILE_2 = 1, 
    FILE_3 = 0, 
    FILE_SIZE = 3,
};

enum Rank : int {
    RANK_1 = 0,
    RANK_2 = 1,
    RANK_3 = 2,
    RANK_4 = 3,
    RANK_SIZE = 4,
};

enum Square : int {
    SQ_BEGIN = 0,
    SQ_11 = 12,
    SQ_12 = 20,
    SQ_13 = 28,
    SQ_14 = 36,
    SQ_21 = 11,
    SQ_22 = 19,
    SQ_23 = 27,
    SQ_24 = 35,
    SQ_31 = 10,
    SQ_32 = 18,
    SQ_33 = 26,
    SQ_34 = 34,
    SQ_END = 48,
    SQ_WALL = -1,
    INC_UP = -8,
    INC_DOWN = 8,
    INC_LEFT = -1,
    INC_RIGHT = 1,
    INC_LEFTUP = -9,
    INC_RIGHTUP = -7,
    INC_LEFTDOWN = 7,
    INC_RIGHTDOWN = 9,
    INC_NONE = 0,
};

enum Piece : int {
    PROM_FLAG = 1 << 3,
    EMPTY = 0,
    HIYOKO = 1,
    KIRIN = 2,
    ZOU = 3,
    LION = 4, 
    NIWATORI = HIYOKO | PROM_FLAG,
    PIECE_END = NIWATORI + 1,
};

enum ColorPiece : int {
    BLACK_FLAG = 1 << 5,
    WHITE_FLAG = 1 << 6,
    COLOR_WALL_FLAG = 1 << 7,
    UP_FLAG = 1 << 8,
    DOWN_FLAG = 1 << 9,
    LEFT_FLAG = 1 << 10,
    RIGHT_FLAG = 1 << 11,
    LEFTUP_FLAG = 1 << 12,
    LEFTDOWN_FLAG = 1 << 13,
    RIGHTUP_FLAG = 1 << 14,
    RIGHTDOWN_FLAG = 1 << 15,
    BLACK_HIYOKO = HIYOKO | BLACK_FLAG | UP_FLAG, 
    BLACK_KIRIN = KIRIN | BLACK_FLAG | UP_FLAG | DOWN_FLAG | LEFT_FLAG | RIGHT_FLAG, 
    BLACK_ZOU = ZOU | BLACK_FLAG | LEFTUP_FLAG | LEFTDOWN_FLAG | RIGHTUP_FLAG | RIGHTDOWN_FLAG, 
    BLACK_LION = LION | BLACK_FLAG | UP_FLAG | DOWN_FLAG | LEFT_FLAG | RIGHT_FLAG | LEFTUP_FLAG | LEFTDOWN_FLAG | RIGHTUP_FLAG | RIGHTDOWN_FLAG, 
    BLACK_NIWATORI = NIWATORI | BLACK_FLAG | UP_FLAG | DOWN_FLAG | LEFT_FLAG | RIGHT_FLAG | LEFTUP_FLAG | RIGHTUP_FLAG, 
    WHITE_HIYOKO = HIYOKO | WHITE_FLAG | DOWN_FLAG, 
    WHITE_KIRIN = KIRIN | WHITE_FLAG | UP_FLAG | DOWN_FLAG | LEFT_FLAG | RIGHT_FLAG,  
    WHITE_ZOU = ZOU | WHITE_FLAG | LEFTUP_FLAG | LEFTDOWN_FLAG | RIGHTUP_FLAG | RIGHTDOWN_FLAG, 
    WHITE_LION = LION | WHITE_FLAG | UP_FLAG | DOWN_FLAG | LEFT_FLAG | RIGHT_FLAG | LEFTUP_FLAG | LEFTDOWN_FLAG | RIGHTUP_FLAG | RIGHTDOWN_FLAG,  
    WHITE_NIWATORI = NIWATORI | WHITE_FLAG | UP_FLAG | DOWN_FLAG | LEFT_FLAG | RIGHT_FLAG | LEFTDOWN_FLAG | RIGHTDOWN_FLAG, 
    COLOR_EMPTY = EMPTY,
    COLOR_WALL = COLOR_WALL_FLAG,
    COLOR_PIECE_END = WHITE_LION + 1,

    CE = COLOR_EMPTY,//短縮形
    BH = BLACK_HIYOKO,
    BZ = BLACK_ZOU,
    BK = BLACK_KIRIN,
    BN = BLACK_NIWATORI,
    BL = BLACK_LION,
    WH = WHITE_HIYOKO,
    WZ = WHITE_ZOU,
    WK = WHITE_KIRIN,
    WN = WHITE_NIWATORI,
    WL = WHITE_LION,
};

enum Hand : int { HAND_NONE = 0 };

Color& operator++(Color& org) {
  org = static_cast<Color>(org + 1);
  return org;
}

Piece& operator++(Piece& org) {
  org = static_cast<Piece>(org + 1);
  return org;
}

Square operator+(Square l, Square r) {
  return static_cast<Square>(static_cast<int>(l) + static_cast<int>(r));
}
Square operator-(Square l, Square r) {
  return static_cast<Square>(static_cast<int>(l) - static_cast<int>(r));
}

constexpr inline int HAND_SHIFT[PIECE_END] = {0, 0, 3, 6, 0, 0};
constexpr inline int HAND_INC[PIECE_END] = {0, 1 << 0, 1 << 3, 1 << 6, 0, 0 };
constexpr inline int HAND_MASK[PIECE_END] = {0, 0x3 << 0, 0x3 << 3, 0x3 << 6, 0, 0 };
constexpr inline Square SQUARE_INDEX[SQUARE_SIZE+1] = { 
    SQ_31, SQ_21, SQ_11,
    SQ_32, SQ_22, SQ_12,
    SQ_33, SQ_23, SQ_13,
    SQ_34, SQ_24, SQ_14, SQ_WALL
};

extern int g_color_piece_no[COLOR_PIECE_END]; 
extern int g_piece_no_color_piece[12];
extern int g_piece_color_piece[COLOR_SIZE][12];

constexpr inline int DELTA_OFFSET = 46;
constexpr inline int DELTA_NB = DELTA_OFFSET * 2 + 1;

extern Square g_delta_inc_all[DELTA_NB];
extern ColorPiece g_delta_mask[DELTA_NB];

inline Square delta_inc_all(const Square delta) {
    ASSERT(delta + DELTA_OFFSET >= 0);
    ASSERT(delta + DELTA_OFFSET < DELTA_NB);
    return g_delta_inc_all[DELTA_OFFSET + delta];
}

inline ColorPiece delta_mask(const Square delta) {
    ASSERT2(delta + DELTA_OFFSET >= 0,{ Tee<<delta<<std::endl; });
    ASSERT2(delta + DELTA_OFFSET < DELTA_NB,{Tee<<delta<<std::endl;});
    return g_delta_mask[DELTA_OFFSET + delta];
}

inline bool pseudo_attack(const ColorPiece p, const Square delta) {
    return (delta_mask(delta) & p) != 0;
}

inline std::string color_str(const Color c) {
    return c == BLACK ? "先手" : "後手";
}
std::ostream& operator<<(std::ostream& os, const Color c) {
    os << color_str(c);
    return os;
}
inline constexpr Color change_turn(const Color turn) {
    return static_cast<Color>(static_cast<int>(turn) ^ 1);
}

inline constexpr Square square(const File f, const Rank r) {
    return static_cast<Square>(f + (r * 8) + 10);
}

inline constexpr Rank sq_rank(const Square sq) {
    return static_cast<Rank>((sq-10) >> 3);
}

inline constexpr File sq_file(const Square sq) {
    return static_cast<File>((sq-10) & 7);
}

inline constexpr Hand inc_hand(const Hand h, const Piece p) {
    return static_cast<Hand>(h + HAND_INC[p]);
}

inline constexpr Hand dec_hand(const Hand h, const Piece p) {
    return static_cast<Hand>(h - HAND_INC[p]);
}

inline constexpr bool has_piece(const Hand h, const Piece p) {
    return (h & HAND_MASK[p]) != 0;
}

inline constexpr int num_piece(const Hand h, const Piece p) {
    return (h >> HAND_SHIFT[p]) & 0x3;
}

inline constexpr Move move(const Square from, const Square to, bool prom = false) {
    return static_cast<Move>(
        (static_cast<int>(prom) << 12) | (from << 6) | to
    );
}

inline constexpr Move move(const Square to, const Piece p) {
    return static_cast<Move>(
       (1 << 13) | (p << 6) | (to)
    );
}

inline constexpr Square move_from(const Move m) {
    return static_cast<Square>((m >> 6) & 0x3F);
}

inline constexpr Square move_to(const Move m) {
    return static_cast<Square>(m & 0x3F);
}

inline constexpr Piece move_piece(const Move m) {
    return static_cast<Piece>((m >> 6) & 0x1F);
}

inline constexpr bool move_is_drop(const Move m) {
    return (m & (1 << 13)) != 0;
}

inline constexpr bool move_is_prom(const Move m) {
    return (m & (1 << 12)) != 0;
}

inline ColorPiece color_piece(const Piece p, const Color c) {
    return static_cast<ColorPiece>(g_piece_color_piece[c][p]);
}

inline constexpr Piece to_piece(const ColorPiece p) {
    return static_cast<Piece>(p & 0x1F);
}

inline constexpr Piece prom(const Piece p) {
    return static_cast<Piece>(p | PROM_FLAG);
}

inline constexpr Piece unprom(const Piece p) {
    return static_cast<Piece>(p & (~PROM_FLAG));
}

inline constexpr Color piece_color(const ColorPiece p) {
    return static_cast<Color>((p >> 6) & 0x1);
}

inline bool sq_is_ok(const Square sq) {
    for(auto *p = SQUARE_INDEX; *p != SQ_WALL; ++p) {
        if (*p == sq) {
            return true;
        }
    }
    return false;
}

inline bool piece_is_ok(const Piece p) {
    return (p == HIYOKO || 
           p == KIRIN || 
           p == ZOU || 
           p == LION || 
           p == NIWATORI);
}

inline bool color_is_ok(const Color c) {
    return (c == BLACK || c == WHITE);
}

inline bool move_is_ok(const Move m) {
    if (move_is_drop(m)) {
        const auto piece = move_piece(m);
        const auto to = move_to(m);
        return piece_is_ok(piece) && sq_is_ok(to);
    } else {
        const auto from = move_from(m);
        const auto to = move_to(m);
        return sq_is_ok(from) && sq_is_ok(to);
    }
}

inline bool hand_is_ok(const Hand h) {
    static constexpr Hand HAND_MAX = static_cast<Hand>(219);
    return h >= HAND_NONE && h <= HAND_MAX;
}

inline std::string PIECE_STR[PIECE_END] = {"・","ひ","き","ぞ","ら","x","x","x","x","に"};

inline std::string piece_str(const Piece p) {
    return PIECE_STR[p];
}

inline std::string hand_str(const Hand h) {
    std::string ret = "";
    REP(i, num_piece(h, HIYOKO)) {
        ret += piece_str(HIYOKO);
    }
    REP(i, num_piece(h, KIRIN)) {
        ret += piece_str(KIRIN);
    }
    REP(i, num_piece(h, ZOU)) {
        ret += piece_str(ZOU);
    }
    if (ret == "") {
        return "持ち駒なし";
    } else {
        return ret;
    }
}


inline std::string sq_str(const Square sq) {
    switch (sq) {
        case SQ_11: return "１一";
        case SQ_12: return "１二";
        case SQ_13: return "１三";
        case SQ_14: return "１四";
        case SQ_21: return "２一";
        case SQ_22: return "２二";
        case SQ_23: return "２三";
        case SQ_24: return "２四";
        case SQ_31: return "３一";
        case SQ_32: return "３二";
        case SQ_33: return "３三";
        case SQ_34: return "３四";
        default: return "UNKNOWN_SQ";
    }
}

inline std::string move_str(const Move m) {
    std::string ret = "";
    if (move_is_drop(m)) {
        ret +="打:" + sq_str(move_to(m))+":"+piece_str(move_piece(m));
    } else {
        ret +="from:"+ sq_str(move_from(m)) + " to:" + sq_str(move_to(m)) +" prom:"+to_string(move_is_prom(m));
    }
    return ret;
}

inline NNScore to_nnscore(const float sc) {
    auto score = static_cast<int>(sc * 10000);
    if (score >= 10000) {
        score = 9999;
    } else if (score <= -10000) {
        score = -9999;
    }
    return static_cast<NNScore>(static_cast<double>(score) / 10000.0);
}

inline int color_piece_no(const ColorPiece p) {
    return g_color_piece_no[p];
}

inline ColorPiece piece_no_color_piece(const int index) {
    return static_cast<ColorPiece>(g_piece_no_color_piece[index]);
}

void init_table() {

    REP(i, COLOR_PIECE_END) {
        g_color_piece_no[i] = -1;
    }
    REP(i, 12) {
        g_piece_no_color_piece[i] = -1;
        g_piece_color_piece[BLACK][i] = g_piece_color_piece[WHITE][i] = -1;
    }
    auto n = 0;
    g_color_piece_no[COLOR_EMPTY] = ++n;

    g_color_piece_no[BLACK_HIYOKO] = ++n;
    g_color_piece_no[BLACK_KIRIN] = ++n;
    g_color_piece_no[BLACK_ZOU] = ++n;
    g_color_piece_no[BLACK_LION] = ++n;
    g_color_piece_no[BLACK_NIWATORI] = ++n;
    
    g_color_piece_no[WHITE_HIYOKO] = ++n;
    g_color_piece_no[WHITE_KIRIN] = ++n;
    g_color_piece_no[WHITE_ZOU] = ++n;
    g_color_piece_no[WHITE_LION] = ++n;
    g_color_piece_no[WHITE_NIWATORI] = ++n;

    g_piece_no_color_piece[g_color_piece_no[COLOR_EMPTY]] = COLOR_EMPTY;

    g_piece_no_color_piece[g_color_piece_no[BLACK_HIYOKO]] = BLACK_HIYOKO;
    g_piece_no_color_piece[g_color_piece_no[BLACK_KIRIN]] = BLACK_KIRIN;
    g_piece_no_color_piece[g_color_piece_no[BLACK_ZOU]] = BLACK_ZOU;
    g_piece_no_color_piece[g_color_piece_no[BLACK_LION]] = BLACK_LION;
    g_piece_no_color_piece[g_color_piece_no[BLACK_NIWATORI]] = BLACK_NIWATORI;

    g_piece_no_color_piece[g_color_piece_no[WHITE_HIYOKO]] = WHITE_HIYOKO;
    g_piece_no_color_piece[g_color_piece_no[WHITE_KIRIN]] = WHITE_KIRIN;
    g_piece_no_color_piece[g_color_piece_no[WHITE_ZOU]] = WHITE_ZOU;
    g_piece_no_color_piece[g_color_piece_no[WHITE_LION]] = WHITE_LION;
    g_piece_no_color_piece[g_color_piece_no[WHITE_NIWATORI]] = WHITE_NIWATORI;

    g_piece_color_piece[BLACK][HIYOKO] = BLACK_HIYOKO;
    g_piece_color_piece[BLACK][KIRIN] = BLACK_KIRIN;
    g_piece_color_piece[BLACK][ZOU] = BLACK_ZOU;
    g_piece_color_piece[BLACK][LION] = BLACK_LION;
    g_piece_color_piece[BLACK][NIWATORI] = BLACK_NIWATORI;

    g_piece_color_piece[WHITE][HIYOKO] = WHITE_HIYOKO;
    g_piece_color_piece[WHITE][KIRIN] = WHITE_KIRIN;
    g_piece_color_piece[WHITE][ZOU] = WHITE_ZOU;
    g_piece_color_piece[WHITE][LION] = WHITE_LION;
    g_piece_color_piece[WHITE][NIWATORI] = WHITE_NIWATORI;

    // for(auto *fp = SQUARE_INDEX; *fp != SQ_WALL; ++fp) {
    //     const auto from = *fp;
    //     for(auto *tp = SQUARE_INDEX; *tp != SQ_WALL; ++tp) {
    //         const auto to = *tp;
    //         if (from == to)
    //             continue;
    //         Tee<<from<<":"<<to<<":"<<from-to<<":"<<to-from<<std::endl;
    //     }
    // }
    REP(i, DELTA_NB) {
        g_delta_inc_all[i] = INC_NONE;
        g_delta_mask[i] = COLOR_EMPTY;
    }
    g_delta_mask[DELTA_OFFSET + INC_DOWN] = DOWN_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_UP] = UP_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_LEFT] = LEFT_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_RIGHT] = RIGHT_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_LEFTUP] = LEFTUP_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_LEFTDOWN] = LEFTDOWN_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_RIGHTUP] = RIGHTUP_FLAG;
    g_delta_mask[DELTA_OFFSET + INC_RIGHTDOWN] = RIGHTDOWN_FLAG;

    g_delta_inc_all[DELTA_OFFSET + INC_DOWN] = INC_DOWN;
    g_delta_inc_all[DELTA_OFFSET + INC_UP] = INC_UP;
    g_delta_inc_all[DELTA_OFFSET + INC_LEFT] = INC_LEFT;
    g_delta_inc_all[DELTA_OFFSET + INC_RIGHT] = INC_RIGHT;
    g_delta_inc_all[DELTA_OFFSET + INC_LEFTUP] = INC_LEFTUP;
    g_delta_inc_all[DELTA_OFFSET + INC_LEFTDOWN] = INC_LEFTDOWN;
    g_delta_inc_all[DELTA_OFFSET + INC_RIGHTUP] = INC_RIGHTUP;
    g_delta_inc_all[DELTA_OFFSET + INC_RIGHTDOWN] = INC_RIGHTDOWN;
    

}

#endif