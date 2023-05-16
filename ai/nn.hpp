#ifndef __NN_HPP__
#define __NN_HPP__
#include "game.hpp"
#include "attack.hpp"
#include "common.hpp"
#include <vector>
namespace nn {
constexpr inline int FEAT_SIZE = 42;
typedef std::vector<std::vector<int>> Feature;
typedef double NNScore;

inline NNScore to_nnscore(const float sc) {
    auto score = static_cast<int>(sc * 10000);
    if (score >= 10000) {
        score = 9999;
    } else if (score <= -10000) {
        score = -9999;
    }
    return static_cast<NNScore>(static_cast<double>(score) / 10000.0);
}

Feature feature(const game::Position &pos) {
    Feature feat(FEAT_SIZE, std::vector<int>(SQUARE_SIZE, 0));
    const auto me = pos.turn();
    const auto opp = change_turn(me);

#define CALC_FEAT(piece, turn, feat_index) do {\
    static_assert(feat_index < FEAT_SIZE,"");\
    REP(i, pos.piece_list_size(turn, piece)) {\
        const auto from = (turn == BLACK) ? pos.piece_list(BLACK,piece,i) : invert_sq(pos.piece_list(WHITE, piece,i));\
        const auto index = sq_to_index(from);\
        feat[feat_index][index] = 1;\
    }\
}while(false)

#define CALC_FEAT_LION(turn, feat_index) do {\
    static_assert(feat_index < FEAT_SIZE,"");\
    const auto from = (turn == BLACK) ? pos.king_sq(BLACK) : invert_sq(pos.king_sq(WHITE));\
    const auto index = sq_to_index(from);\
    feat[feat_index][index] = 1;\
}while(false)

#define CALC_FEAT_HAND(piece, turn, feat_index1, feat_index2) do {\
    static_assert(feat_index1 < FEAT_SIZE,"");\
    static_assert(feat_index2 < FEAT_SIZE,"");\
    switch (num_piece(pos.hand(turn),piece)) {\
        case 0: break;\
        case 1: {\
            REP(i, SQUARE_SIZE) {\
                feat[feat_index1][i] = 1;\
            }\
            break;\
        }\
        case 2: {\
            REP(i, SQUARE_SIZE) {\
                feat[feat_index1][i] = 1;\
                feat[feat_index2][i] = 1;\
            }\
            break;\
        }\
        default: assert(false); break;\
    }\
} while(false)

    CALC_FEAT(HIYOKO, me, 0);
    CALC_FEAT(ZOU, me, 1);
    CALC_FEAT(KIRIN, me, 2);
    CALC_FEAT(NIWATORI, me, 3);
    
    CALC_FEAT(HIYOKO, opp, 4);
    CALC_FEAT(ZOU, opp, 5);
    CALC_FEAT(KIRIN, opp, 6);
    CALC_FEAT(NIWATORI, opp, 7);
    
    CALC_FEAT_LION(me, 8);
    CALC_FEAT_LION(opp, 9);

    CALC_FEAT_HAND(HIYOKO, me, 10, 11);
    CALC_FEAT_HAND(ZOU, me, 12, 13);
    CALC_FEAT_HAND(KIRIN, me, 14, 15);

    CALC_FEAT_HAND(HIYOKO, opp, 16, 17);
    CALC_FEAT_HAND(ZOU, opp, 18, 19);
    CALC_FEAT_HAND(KIRIN, opp, 20, 21);

    attack::Attacker attacker(pos);

#define CALC_ATTACK_NUM_FEAT(col, num, index, inv_index, feat_index) do {\
    static_assert(feat_index < FEAT_SIZE,"");\
    if (attacker.attacker_num[col][index] >= num) {\
        feat[feat_index][inv_index] = 1;\
    }\
} while(false)

#define CALC_ATTACK_PIECE_FEAT(col, index, inv_index,piece, feat_index) do {\
    static_assert(feat_index < FEAT_SIZE,"");\
    if (attacker.attacker_piece_num[col][index][piece] != 0) {\
        feat[feat_index][inv_index] = 1;\
    }\
} while (false)

    REP(index, SQUARE_SIZE) {
        const auto me_index = (me == BLACK) ? index : 11 - index;
        const auto opp_index = (opp == BLACK) ? index : 11 - index;
    //利きが何個利いているか
        CALC_ATTACK_NUM_FEAT(me, 1, index, me_index, 22);        
        CALC_ATTACK_NUM_FEAT(me, 2, index,me_index, 23);        
        CALC_ATTACK_NUM_FEAT(me, 3, index,me_index, 24);     
        CALC_ATTACK_NUM_FEAT(me, 4, index,me_index, 25);     
        CALC_ATTACK_NUM_FEAT(opp, 1, index,opp_index, 26);   
        CALC_ATTACK_NUM_FEAT(opp, 2, index,opp_index, 27);   
        CALC_ATTACK_NUM_FEAT(opp, 3, index,opp_index, 28);   
        CALC_ATTACK_NUM_FEAT(opp, 4, index,opp_index, 29);   
    //どの駒が利いているか？
        CALC_ATTACK_PIECE_FEAT(me, index, me_index, HIYOKO, 30);
        CALC_ATTACK_PIECE_FEAT(me, index, me_index, KIRIN, 31);
        CALC_ATTACK_PIECE_FEAT(me, index, me_index, ZOU, 32);
        CALC_ATTACK_PIECE_FEAT(me, index, me_index, LION, 33);
        CALC_ATTACK_PIECE_FEAT(me, index, me_index, NIWATORI, 34);
        CALC_ATTACK_PIECE_FEAT(opp, index, opp_index, HIYOKO, 35);
        CALC_ATTACK_PIECE_FEAT(opp, index, opp_index, KIRIN, 36);
        CALC_ATTACK_PIECE_FEAT(opp, index, opp_index, ZOU, 37);
        CALC_ATTACK_PIECE_FEAT(opp, index, opp_index, LION, 38);
        CALC_ATTACK_PIECE_FEAT(opp, index, opp_index, NIWATORI, 39);
    }
    //王手されているかどうか
    const auto me_lion_sq = sq_to_index(pos.king_sq(me));
    if (attacker.attacker_num[opp][me_lion_sq] != 0) {
        REP(i, SQUARE_SIZE) {
            feat[40][i] = 1;
        }
    }
    const auto opp_lion_sq = sq_to_index(pos.king_sq(opp));
    if (attacker.attacker_num[me][opp_lion_sq] != 0) {
        REP(i, SQUARE_SIZE) {
            feat[41][i] = 1;
        }
    }

#undef CALC_FEAT
#undef CALC_FEAT_LION
#undef CALC_FEAT_HAND
#undef CALC_ATTACK_NUM_FEAT
#undef CALC_ATTACK_PIECE_FEAT
    return feat;
}

void test_nn() {
    ColorPiece pieces[SQUARE_SIZE] = {
        WK, WL, WZ,
        CE, WH, CE,
        CE, BH, CE,
        BZ, BL, BK
    };
    Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
    game::Position pos(pieces,hand,BLACK);
    Feature f = feature(pos);
    REP(i, FEAT_SIZE) {
        Tee<<"index:"<<i<<std::endl;
        REP(rank, 4) {
            REP(file, 3) {
                const auto sq = file + rank *3;
                Tee<<f[i][sq];
            }
            Tee<<"\n";
        }
    }
}

}
#endif