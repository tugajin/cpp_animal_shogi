#ifndef __ORACLE_HPP__
#define __ORACLE_HPP__
#include "game.hpp"
#include "hash.hpp"
#include <vector>
#include <algorithm>


namespace oracle {
class OracleData {
private:
    std::vector<uint64> win_data;
    std::vector<uint64> lose_data;
    std::vector<uint64> draw_data;
    int result(const uint64 index) const {
        if (std::binary_search(win_data.begin(), win_data.end(),index)) {
            return 1;
        }
        if (std::binary_search(lose_data.begin(), lose_data.end(),index)) {
            return -1;
        }
        if (std::binary_search(draw_data.begin(), draw_data.end(),index)) {
            return 0;
        }
        // こないはず
        return 99;
    }
public:
    static constexpr int win_size = 196773087;
    static constexpr int lose_size = 47347380;
    static constexpr int draw_size = 2682700;
    OracleData() {
        
    }
    void load() {
        win_data.resize(OracleData::win_size);
        lose_data.resize(OracleData::lose_size);
        draw_data.resize(OracleData::draw_size);
        std::ifstream ifs_win("/home/tugajin/Documents/cpp_animal_shogi/data/win.dat", std::ios_base::in | std::ios_base::binary);
        std::ifstream ifs_lose("/home/tugajin/Documents/cpp_animal_shogi/data/lose.dat", std::ios_base::in | std::ios_base::binary);
        std::ifstream ifs_draw("/home/tugajin/Documents/cpp_animal_shogi/data/draw.dat", std::ios_base::in | std::ios_base::binary);
        ifs_win.read((char*)&win_data[0], OracleData::win_size * sizeof(win_data[0]));
        ifs_lose.read((char*)&lose_data[0], OracleData::lose_size * sizeof(lose_data[0]));
        ifs_draw.read((char*)&draw_data[0], OracleData::draw_size * sizeof(draw_data[0]));
        ifs_win.close();
        ifs_lose.close();
        ifs_draw.close();
    }
    int result(const game::Position &pos)const;
    uint64 win(const int index) const {
        return win_data[index];
    }
    uint64 lose(const int index) const {
        return lose_data[index];
    }
    uint64 draw(const int index) {
        return draw_data[index];
    }
};
int OracleData::result(const game::Position &pos) const {
    if (pos.turn() == BLACK) {
        const auto mirror_pos = pos.mirror();
        const auto key0 = hash::hash_key(pos);
        const auto key1 = hash::hash_key(mirror_pos);
        auto result = this->result(static_cast<uint64>(key0));
        if (result != 99) { return result; }
        result = this->result(static_cast<uint64>(key1));
        if (result != 99) { return result; }
    } else {
        const auto rotate_pos = pos.rotate();
        const auto mirror_rotate_pos = rotate_pos.mirror();
        const auto key2 = hash::hash_key(rotate_pos);
        const auto key3 = hash::hash_key(mirror_rotate_pos);
        auto result = this->result(static_cast<uint64>(key2));
        if (result != 99) { return result; }
        result = this->result(static_cast<uint64>(key3));
        if (result != 99) { return result; }
    }
    return 99;
}
extern OracleData g_oracle;
}
#endif