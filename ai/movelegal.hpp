#ifndef __MOVELEGAL_HPP__
#define __MOVELEGAL_HPP__

#include "game.hpp"
#include "movelist.hpp"
#include "attack.hpp"
#include "movegen.hpp"
#include "movedrop.hpp"
#include "moveevasion.hpp"
#include "movecheck.hpp"
#include "matesearch.hpp"
#include "hash.hpp"

#include <unordered_map>

namespace gen {
template<bool is_exists = false> bool legal_moves(const game::Position &pos, movelist::MoveList &ml) {
    auto result = false;
    if (attack::in_checked(pos)) {
        result = evasion_moves<is_exists>(pos, ml);
#if DEBUG
    if (!is_exists) {
        movelist::MoveList all_ml;
        pos_moves(pos, all_ml);
        REP(i, ml.len()) {
            auto next = pos.next(ml[i]);
            ASSERT2(!next.is_win(),{
                Tee<<pos<<std::endl;
                Tee<<next<<std::endl;
                Tee<<ml<<std::endl;
            });
        }
        movelist::MoveList tmp;
        REP(i, all_ml.len()) {
            auto next = pos.next(all_ml[i]);
            if (!next.is_win()) {
                tmp.add(all_ml[i]);
            }
        }
        ASSERT2(tmp.len() == ml.len(),{
            Tee<<pos<<std::endl;
            Tee<<tmp<<std::endl;
            Tee<<ml<<std::endl;
        })
    } else {
        movelist::MoveList tmp;
        evasion_moves(pos, tmp);
        ASSERT2(result == (tmp.len() !=0),{
            Tee<<pos<<std::endl;
            Tee<<tmp<<std::endl;
            Tee<<result<<std::endl;
        });
    }
#endif
    } else {
        result = pos_moves<is_exists>(pos, ml);
        result |= drop_moves<is_exists>(pos, ml);
#if DEBUG
        movelist::MoveList check_ml;
        pos_check_moves(pos, check_ml);
        drop_check_moves(pos, check_ml);
    if (!is_exists) {
        REP(i, ml.len()) {
            auto next = pos.next(ml[i]);
            if(attack::in_checked(next)) {
                if (!check_ml.contain(ml[i])) {
                    Tee<<pos<<std::endl;
                    Tee<<check_ml<<std::endl;
                    Tee<<move_str(ml[i])<<std::endl;
                    ASSERT(false);
                }
            }
        }
    } else {
        movelist::MoveList tmp;
        pos_moves(pos, tmp);
        drop_moves(pos, tmp);
        ASSERT(result == (tmp.len() !=0));
    }
#endif
    }
    return result;
}

bool has_legal(const game::Position &pos) {
    movelist::MoveList dummy;
    return legal_moves<true>(pos, dummy);
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
void test_gen2() {
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_LION, COLOR_EMPTY,
        };
        game::Position pos(pieces,hand,BLACK);
        movelist::MoveList ml;
        legal_moves(pos,ml);
        REP(i, ml.len()) {
            auto pos2 = pos.next(ml[i]);
            Tee<<pos2<<std::endl;
        }
    }
    Tee<<"--------------------\n";
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            COLOR_EMPTY, WHITE_ZOU, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_HIYOKO, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_LION, WHITE_ZOU,
        };
        game::Position pos(pieces,hand,BLACK);
        movelist::MoveList ml;
        legal_moves(pos,ml);
        REP(i, ml.len()) {
            auto pos2 = pos.next(ml[i]);
            Tee<<move_str(ml[i])<<std::endl;
            Tee<<pos2<<std::endl;
        }
    }
    Tee<<"--------------------\n";
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            BLACK_HIYOKO, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_LION, COLOR_EMPTY,
        };
        game::Position pos(pieces,hand,BLACK);
        movelist::MoveList ml;
        legal_moves(pos,ml);
        REP(i, ml.len()) {
            auto pos2 = pos.next(ml[i]);
            Tee<<move_str(ml[i])<<std::endl;
            Tee<<pos2<<std::endl;
        }
    }
    Tee<<"--------------------\n";
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN, WHITE_LION, COLOR_EMPTY,
            BLACK_HIYOKO, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_LION, COLOR_EMPTY,
        };
        game::Position pos(pieces,hand,BLACK);
        movelist::MoveList ml;
        legal_moves(pos,ml);
        REP(i, ml.len()) {
            auto pos2 = pos.next(ml[i]);
            Tee<<move_str(ml[i])<<std::endl;
            Tee<<pos2<<std::endl;
        }
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_NIWATORI, WHITE_LION, COLOR_EMPTY,
            BLACK_HIYOKO, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_LION, COLOR_EMPTY,
        };
        game::Position pos(pieces,hand,BLACK);
        movelist::MoveList ml;
        legal_moves(pos,ml);
        REP(i, ml.len()) {
            auto pos2 = pos.next(ml[i]);
            Tee<<move_str(ml[i])<<std::endl;
            Tee<<pos2<<std::endl;
        }
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, WHITE_ZOU,
            WHITE_KIRIN, WHITE_HIYOKO, COLOR_EMPTY,
            BLACK_LION, BLACK_HIYOKO, COLOR_EMPTY,
            BLACK_ZOU, COLOR_EMPTY, BLACK_KIRIN,
        };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        legal_moves(pos,ml);
        REP(i, ml.len()) {
            auto pos2 = pos.next(ml[i]);
            Tee<<move_str(ml[i])<<std::endl;
            Tee<<pos2<<std::endl;
        }
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, WHITE_ZOU,
            WHITE_KIRIN, WHITE_HIYOKO, BLACK_ZOU,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            BLACK_KIRIN, WHITE_HIYOKO, BLACK_LION,
        };
        game::Position pos(pieces,hand,WHITE);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        legal_moves(pos,ml);
        ASSERT(ml.len() != 0);
        Tee<<ml<<std::endl;
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        hand[BLACK] = inc_hand(hand[BLACK],HIYOKO);
        hand[BLACK] = inc_hand(hand[BLACK],KIRIN);
        hand[BLACK] = inc_hand(hand[BLACK],ZOU);
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_LION,
        };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        drop_check_moves(pos,ml);
        ASSERT(ml.len() == 9);
        Tee<<ml<<std::endl;
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_KIRIN,
            COLOR_EMPTY, BLACK_HIYOKO, BLACK_LION,
        };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        pos_check_moves(pos,ml);
        ASSERT(ml.len() != 0);
        Tee<<ml<<std::endl;
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_HIYOKO,
            COLOR_EMPTY, BLACK_ZOU, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_LION,
        };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        pos_check_moves(pos,ml);
        ASSERT(ml.len() == 2);
        Tee<<ml<<std::endl;
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, WHITE_LION, COLOR_EMPTY,
            BLACK_HIYOKO, COLOR_EMPTY, BLACK_HIYOKO,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_LION,
        };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        pos_check_moves(pos,ml);
        ASSERT(ml.len() == 2);
        Tee<<ml<<std::endl;
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            BLACK_HIYOKO, WHITE_LION, BLACK_HIYOKO,
            COLOR_EMPTY, COLOR_EMPTY, COLOR_EMPTY,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_LION,
        };
        game::Position pos(pieces,hand,BLACK);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        pos_check_moves(pos,ml);
        Tee<<ml<<std::endl;
        ASSERT(ml.len() == 0);
    }
    {
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        ColorPiece pieces[SQUARE_SIZE] = {
            COLOR_EMPTY, COLOR_EMPTY, BLACK_KIRIN,
            WHITE_LION, BLACK_ZOU, COLOR_EMPTY,
            WHITE_NIWATORI, BLACK_HIYOKO, BLACK_KIRIN,
            COLOR_EMPTY, COLOR_EMPTY, BLACK_LION,
        };
        game::Position pos(pieces,hand,WHITE);
        Tee<<pos<<std::endl;
        movelist::MoveList ml;
        pos_check_moves(pos,ml);
        Tee<<ml<<std::endl;
        ASSERT(ml.len() == 2);
    }
}
void test_gen3() {
    
    std::unordered_map<Key, int> key_dict;
    uint64 mate_num = 0;
    uint64 draw_num = 0;
    REP(i, 999999999999) {
        Tee<<i<<":"<<key_dict.size()<<" mate:"<<mate_num<<" draw:"<<draw_num<<"\r";
        ColorPiece pieces[SQUARE_SIZE] = {
            WHITE_KIRIN, WHITE_LION, WHITE_ZOU,
            COLOR_EMPTY, WHITE_HIYOKO, COLOR_EMPTY,
            COLOR_EMPTY, BLACK_HIYOKO, COLOR_EMPTY,
            BLACK_ZOU, BLACK_LION, BLACK_KIRIN,
        };
        Hand hand[COLOR_SIZE] = { HAND_NONE, HAND_NONE };
        game::Position pos(pieces,hand,BLACK);
        while(true) {
            const auto key = hash::hash_key(pos);
            const auto new_pos = hash::from_hash(key);
            ASSERT2(key == hash::hash_key(new_pos),{
                Tee<<pos<<std::endl;
                Tee<<new_pos<<std::endl;
            });
            if (key_dict.count(hash::hash_key(pos)) == 0){
                key_dict.insert({hash::hash_key(pos), 1});
            }
            if (pos.is_draw()) {
                //Tee<<pos<<std::endl;
                draw_num++;
                break;
            }
            if (pos.is_lose()) {
                //Tee<<pos<<std::endl;
                mate_num++;
                break;
            }
            if (pos.is_win()) {
                //Tee<<pos<<std::endl;
                mate_num++;
                break;
            }
            auto ml = movelist::MoveList();
            legal_moves(pos, ml);
            if (ml.len() == 0) {
                //Tee<<pos<<std::endl;
                mate_num++;
                break;
            }
            auto mv = ml[my_rand(ml.len())];
            if (!attack::in_checked(pos)) {
                const auto mate = mate::mate_search(pos,5);
                if (mate != MOVE_NONE) {
                    //Tee<<pos<<std::endl;
                    //Tee<<move_str(mate)<<std::endl;
                    mv = mate;
                }
            }
            //Tee<<pos<<std::endl;
            //Tee<<move_str(mv)<<std::endl;
            //Tee<<mv<<std::endl;
            pos = pos.next(mv);
        }
    }
}

}
namespace game {
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
    return !gen::has_legal(*this);
}
}
#endif