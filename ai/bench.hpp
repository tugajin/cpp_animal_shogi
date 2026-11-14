#ifndef __BENCH_HPP__
#define __BENCH_HPP__

#include <vector>
#include <chrono>
#include <iomanip>
#include "common.hpp"
#include "util.hpp"
#include "game.hpp"
#include "hash.hpp"
#include "ubfm.hpp"
#include "model.hpp"
#include "movelist.hpp"
#include "movelegal.hpp"

namespace bench {

struct BenchResult {
    std::string label;
    int thread_num;
    int simulation_num;
    double wall_time_sec;
    int64_t total_nodes;
    double nps;
    int64_t infer_calls;
    int64_t infer_total_batch;
    double avg_batch_size;
    double infer_time_sec;
    double infer_ratio;
};

void print_result(const BenchResult &r) {
    Tee << "=== Bench Result [" << r.label << "] ===" << std::endl;
    Tee << "threads:          " << r.thread_num << std::endl;
    Tee << "simulations:      " << r.simulation_num << std::endl;
    Tee << "wall time:        " << std::fixed << std::setprecision(3) << r.wall_time_sec << " sec" << std::endl;
    Tee << "total nodes:      " << r.total_nodes << std::endl;
    Tee << "NPS:              " << std::fixed << std::setprecision(0) << r.nps << std::endl;
    Tee << "infer calls:      " << r.infer_calls << std::endl;
    Tee << "infer total batch:" << r.infer_total_batch << std::endl;
    Tee << "avg batch size:   " << std::fixed << std::setprecision(2) << r.avg_batch_size << std::endl;
    Tee << "infer time:       " << std::fixed << std::setprecision(3) << r.infer_time_sec << " sec" << std::endl;
    Tee << "infer ratio:      " << std::fixed << std::setprecision(1) << r.infer_ratio * 100.0 << " %" << std::endl;
    Tee << "====================" << std::endl;
}

std::vector<game::Position> make_bench_positions() {
    std::vector<game::Position> positions;

    positions.push_back(hash::hirate());

    {
        auto pos = hash::hirate();
        auto ml = movelist::MoveList();
        gen::legal_moves(pos, ml);
        if (ml.len() > 0) {
            pos = pos.next(ml[0]);
            positions.push_back(pos);
            ml = movelist::MoveList();
            gen::legal_moves(pos, ml);
            if (ml.len() > 0) {
                pos = pos.next(ml[0]);
                positions.push_back(pos);
            }
        }
    }
    {
        auto pos = hash::hirate();
        auto ml = movelist::MoveList();
        gen::legal_moves(pos, ml);
        if (ml.len() > 2) {
            pos = pos.next(ml[2]);
            positions.push_back(pos);
            ml = movelist::MoveList();
            gen::legal_moves(pos, ml);
            if (ml.len() > 1) {
                pos = pos.next(ml[1]);
                positions.push_back(pos);
            }
        }
    }

    {
        auto pos = hash::hirate();
        auto ml = movelist::MoveList();
        gen::legal_moves(pos, ml);
        if (ml.len() > 1) {
            pos = pos.next(ml[1]);
            positions.push_back(pos);
        }
    }

    return positions;
}

BenchResult run_one(const std::string &label, const int thread_num, const int total_simulations, const bool use_batch) {
    auto positions = make_bench_positions();

    model::g_gpu_model[0].stats.reset();

    ubfm::g_use_batch_predict = use_batch;
    if (use_batch) {
        model::g_batch_predictor.start(0, 128, 1000);
    }

    ubfm::g_searcher_global.THREAD_NUM = thread_num;
    ubfm::g_searcher_global.SIMULATION_NUM = total_simulations;
    ubfm::g_searcher_global.init();

    int64_t total_nodes = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (auto &pos : positions) {
        ubfm::g_searcher_global.clear_tree();
        ubfm::g_searcher_global.root_node.pos = pos;
        ubfm::g_searcher_global.run();
        ubfm::g_searcher_global.join();
        total_nodes += ubfm::g_searcher_global.root_node.n;
    }

    auto t1 = std::chrono::steady_clock::now();
    double wall_sec = std::chrono::duration<double>(t1 - t0).count();

    if (use_batch) {
        model::g_batch_predictor.stop();
    }
    ubfm::g_use_batch_predict = false;

    auto &st = model::g_gpu_model[0].stats;
    BenchResult r;
    r.label = label;
    r.thread_num = thread_num;
    r.simulation_num = total_simulations;
    r.wall_time_sec = wall_sec;
    r.total_nodes = total_nodes;
    r.nps = (wall_sec > 0) ? total_nodes / wall_sec : 0;
    r.infer_calls = st.call_count.load();
    r.infer_total_batch = st.total_batch.load();
    r.avg_batch_size = (r.infer_calls > 0) ? static_cast<double>(r.infer_total_batch) / r.infer_calls : 0;
    r.infer_time_sec = st.total_time_us.load() / 1e6;
    r.infer_ratio = (wall_sec > 0) ? r.infer_time_sec / wall_sec : 0;
    return r;
}

void print_summary(const std::vector<BenchResult> &results) {
    Tee << "\n=== Summary ===" << std::endl;
    Tee << std::setw(16) << "mode"
        << std::setw(8) << "threads"
        << std::setw(12) << "wall(s)"
        << std::setw(10) << "NPS"
        << std::setw(10) << "calls"
        << std::setw(10) << "avg_batch"
        << std::setw(12) << "infer(s)"
        << std::setw(10) << "infer%"
        << std::endl;
    for (auto &r : results) {
        Tee << std::setw(16) << r.label
            << std::setw(8) << r.thread_num
            << std::setw(12) << std::fixed << std::setprecision(3) << r.wall_time_sec
            << std::setw(10) << std::fixed << std::setprecision(0) << r.nps
            << std::setw(10) << r.infer_calls
            << std::setw(10) << std::fixed << std::setprecision(2) << r.avg_batch_size
            << std::setw(12) << std::fixed << std::setprecision(3) << r.infer_time_sec
            << std::setw(10) << std::fixed << std::setprecision(1) << r.infer_ratio * 100.0
            << std::endl;
    }
}

void run_bench() {
    constexpr int TOTAL_SIM = 2000;
    const std::vector<int> thread_counts = {1, 2, 4};

    Tee << "\n### UBFM Benchmark ###" << std::endl;
    Tee << "positions: " << make_bench_positions().size() << std::endl;
    Tee << "total simulations: " << TOTAL_SIM << std::endl;
    Tee << std::endl;

    std::vector<BenchResult> results;

    Tee << "========== Non-Batched ==========" << std::endl;
    for (auto th : thread_counts) {
        auto label = "direct_t" + to_string(th);
        Tee << "--- " << label << " ---" << std::endl;
        auto r = run_one(label, th, TOTAL_SIM, false);
        print_result(r);
        results.push_back(r);
    }

    Tee << "\n========== Batched ==========" << std::endl;
    for (auto th : thread_counts) {
        auto label = "batch_t" + to_string(th);
        Tee << "--- " << label << " ---" << std::endl;
        auto r = run_one(label, th, TOTAL_SIM, true);
        print_result(r);
        results.push_back(r);
    }

    print_summary(results);
}

void run_bench_simple() {
    Tee << "\n### Simple Bench (non-batched, 1 thread) ###\n";
    const std::vector<int> sim_counts = {100, 500, 1000, 2000};
    for (auto sim : sim_counts) {
        Tee << "--- sim=" << sim << " ---\n";
        auto r = run_one("test", 1, sim, false);
        Tee << "OK nodes=" << r.total_nodes << " nps=" << std::fixed << std::setprecision(0) << r.nps << "\n";
    }
}

}

#endif
