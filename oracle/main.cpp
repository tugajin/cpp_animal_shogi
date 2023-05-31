#include "dobutsu.h"
#include "allStateTable.h"
#include "winLoseTable.h"
#include "nlohmann/json.hpp"
#include "util.hpp"
#include "common.hpp"
#include "hash.hpp"
#include "game.hpp"
#include <fstream>
#include <vector>

using json = nlohmann::json;

TeeStream Tee;

int g_color_piece_no[COLOR_PIECE_END];
int g_piece_no_color_piece[12];
int g_piece_color_piece[COLOR_SIZE][12];

Square g_delta_inc_all[DELTA_NB];
ColorPiece g_delta_mask[DELTA_NB];

ColorPiece conv_p(char p) {
  switch(p) {
    case (char)Ptype::EMPTY: 
      return COLOR_EMPTY;
    case (char)Ptype::makePtype(WHITE,Ptype::ELEPHANT):
      return WHITE_ZOU;
    case (char)Ptype::makePtype(WHITE,Ptype::LION):
      return WHITE_LION;
    case (char)Ptype::makePtype(WHITE,Ptype::GIRAFFE):
      return WHITE_KIRIN;
    case (char)Ptype::makePtype(WHITE,Ptype::BABY):
      return WHITE_HIYOKO;
    case (char)Ptype::makePtype(WHITE,Ptype::CHICKEN):
      return WHITE_NIWATORI;
    case (char)Ptype::makePtype(BLACK,Ptype::BABY):
      return BLACK_HIYOKO;
    case (char)Ptype::makePtype(BLACK,Ptype::ELEPHANT):
      return BLACK_ZOU;
    case (char)Ptype::makePtype(BLACK,Ptype::LION):
      return BLACK_LION;
    case (char)Ptype::makePtype(BLACK,Ptype::GIRAFFE):
      return BLACK_KIRIN;
    case (char)Ptype::makePtype(BLACK,Ptype::CHICKEN):
      return BLACK_NIWATORI;
  }
  std::cout<<"error\n";
  return COLOR_EMPTY;
}

game::Position conv_state(const State &s) {
  ColorPiece pieces[SQUARE_SIZE] = {COLOR_EMPTY,COLOR_EMPTY,COLOR_EMPTY,
                                    COLOR_EMPTY,COLOR_EMPTY,COLOR_EMPTY,
                                    COLOR_EMPTY,COLOR_EMPTY,COLOR_EMPTY,
                                    COLOR_EMPTY,COLOR_EMPTY,COLOR_EMPTY
                                    };
  for (auto sq = 0; sq < 12; sq++) {
    const auto file = int(sq % 4);
    const auto rank = int(sq / 4);
    const auto sq2 = file *3 + (2-rank);
    pieces[sq2] = conv_p(s.board[sq]);
  }
  Hand hand[2] = {HAND_NONE, HAND_NONE};
  for(auto i = 1; i <= s.stands[0]; i++) {
    hand[BLACK2] = inc_hand(hand[BLACK2],HIYOKO);
  }
  for(auto i = 1; i <= s.stands[1]; i++) {
    hand[BLACK2] = inc_hand(hand[BLACK2],ZOU);
  }
  for(auto i = 1; i <= s.stands[2]; i++) {
    hand[BLACK2] = inc_hand(hand[BLACK2],KIRIN);
  }
  for(auto i = 1; i <= s.stands[3]; i++) {
    hand[WHITE2] = inc_hand(hand[WHITE2],HIYOKO);
  }
  for(auto i = 1; i <= s.stands[4]; i++) {
    hand[WHITE2] = inc_hand(hand[WHITE2],ZOU);
  }
  for(auto i = 1; i <= s.stands[5]; i++) {
    hand[WHITE2] = inc_hand(hand[WHITE2],KIRIN);
  }
  return game::Position(pieces,hand,BLACK2);
}

int main() {

  init_table();
  AllStateTable allS("allstates.dat");     
  WinLoseTable winLose(allS,"winLoss.dat","winLossCount.dat");
  const auto size = allS.size();
  std::ofstream ofs_win("win.dat", std::ios_base::out | std::ios_base::binary);
  std::ofstream ofs_lose("lose.dat", std::ios_base::out | std::ios_base::binary);
  std::ofstream ofs_draw("draw.dat", std::ios_base::out | std::ios_base::binary);
  std::vector<uint64> win_vec;
  std::vector<uint64> lose_vec;
  std::vector<uint64> draw_vec;
  std::cout<<"size:"<<size<<std::endl;
  const auto index = allS.find(72063277088239616);
  std::cout<<index<<std::endl;
  State s(72063277088239616);
  std::cout<<s<<std::endl;
  int dummy = 0;
  std::cout<<winLose.getWinLose(s,dummy)<<std::endl;
  game::Position pos = conv_state(s);
  Tee<<pos.history()<<std::endl;
  // for (size_t i = 0; i < size; i++) {
  //   const auto code = allS[i];
  //   State s(code);
  //   auto dummy = 0;
  //   const auto result = winLose.getWinLose(s,dummy);
  //   game::Position pos = conv_state(s);
  //   const auto key = pos.history();
  //   if (result == 1) {
  //     win_vec.push_back(key);
  //   } else if (result == -1) {
  //     lose_vec.push_back(key);
  //   } else if (result == 0) {
  //     draw_vec.push_back(key);
  //   }
  //   if (i&&i % 1000000 == 0) {
  //     std::cout<<i<<":"<<(i*100)/size<<"\n";
  //     //break;
  //   }
  // }

  std::sort(win_vec.begin(),win_vec.end());
  std::sort(lose_vec.begin(),lose_vec.end());
  std::sort(draw_vec.begin(),draw_vec.end());
  
  ofs_win.write((char*)&win_vec[0],  win_vec.size()  * sizeof(win_vec[0]));
  ofs_lose.write((char*)&lose_vec[0], lose_vec.size() * sizeof(lose_vec[0]));
  ofs_draw.write((char*)&draw_vec[0], draw_vec.size() * sizeof(draw_vec[0]));
  
  std::cout<<win_vec.size()<<std::endl;
  std::cout<<lose_vec.size()<<std::endl;
  std::cout<<draw_vec.size()<<std::endl;

  ofs_win.close();
  ofs_lose.close();
  ofs_draw.close();
  // std::cout<<code_vec[0]<<":"<<static_cast<int>(result_vec[0])<<std::endl;
  // std::ifstream ifs("myallstates.dat", std::ios_base::in | std::ios_base::binary);
  // std::vector<uint64> input_data;
  // input_data.resize(code_vec.size());
  // ifs.read((char*)&input_data[0], code_vec.size() * sizeof(code_vec[0]));
  // std::cout<<input_data.size()<<std::endl;
  // std::cout<<code_vec[0]<<std::endl;
  // std::cout<<static_cast<uint64>(input_data[0])<<std::endl;
  return 0;

}
