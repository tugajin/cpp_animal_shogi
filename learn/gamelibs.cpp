#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include "../ai/common.hpp"
#include "../ai/game.hpp"
#include "../ai/movelegal.hpp"
#include "../ai/hash.hpp"
#include "../ai/nn.hpp"
#include "../ai/matesearch.hpp"

namespace py = pybind11;

TeeStream Tee;
int g_color_piece_no[COLOR_PIECE_END];
int g_piece_no_color_piece[12];
int g_piece_color_piece[COLOR_SIZE][12];

Square g_delta_inc_all[DELTA_NB];
ColorPiece g_delta_mask[DELTA_NB];

PYBIND11_MODULE(gamelibs, m) {
    
    m.doc() = "gamelibs made by pybind11";

    init_table();

    m.def("color_piece_no",&color_piece_no);

    py::enum_<Color>(m, "Color", py::arithmetic())
        .value("BLACK", Color::BLACK)
        .value("WHITE", Color::WHITE);

    py::enum_<ColorPiece>(m, "ColorPiece", py::arithmetic())
        .value("COLOR_EMPTY", ColorPiece::COLOR_EMPTY)
        .value("BLACK_HIYOKO", ColorPiece::BLACK_HIYOKO);

    py::enum_<Move>(m, "Move", py::arithmetic())
        .value("MOVE_NONE", Move::MOVE_NONE);

    m.def("val_to_move",[](int m){
        return Move(m);
    });
    
    m.def("move_to_val",[](Move x){
        return static_cast<int>(x);
    });

    m.def("from_hash",&hash::from_hash);
    m.def("hirate", &hash::hirate);

    m.def("hash_key", &hash::hash_key);
    m.def("legal_moves", &gen::legal_moves);
    m.def("feature", &nn::feature);
    m.def("in_checked",&attack::in_checked);

    
    m.def("mate_search", &mate::mate_search);

    py::class_<movelist::MoveList>(m, "MoveList")
        .def(py::init<>())
        .def("init", &movelist::MoveList::init)
        .def("add", &movelist::MoveList::add)
        .def("begin", &movelist::MoveList::begin)
        .def("end", &movelist::MoveList::end)
        .def("len", &movelist::MoveList::len)
        .def("__str__", &movelist::MoveList::str)
        .def("__getitem__", &movelist::MoveList::operator[]);

    py::class_<game::Position>(m, "Position")
        .def(py::init<>())
        .def("turn", &game::Position::turn)
        .def("next",&game::Position::next)
        .def("__str__",&game::Position::str)
        .def("is_win",&game::Position::is_win)
        .def("is_draw",&game::Position::is_draw)
        .def("is_lose",&game::Position::is_lose)
        .def("is_done",&game::Position::is_done);

}