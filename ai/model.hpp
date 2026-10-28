#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include "util.hpp"
#include "thread.hpp"
#include "oracle.hpp"
#include "hash.hpp"
#include "game.hpp"
#include "nn.hpp"

namespace model {

// ONNX Runtime version of GPUModel.
// Public interface is identical to the previous LibTorch/TorchScript version:
//   load_model(id), predict(feat_list, outputs), g_gpu_model[], predict()
// Model contract (see learn/export_onnx.py):
//   input  "input" float32 (N, 42, 3, 4), batch axis dynamic
//   output "value" float32 (N, 1)
//   opset 17, file ./model/best_single.onnx
class GPUModel {
public:
    GPUModel() : env(ORT_LOGGING_LEVEL_WARNING, "cpp_animal_shogi") {}
    void load_model(const int id);
    void predict(std::vector<nn::Feature> &feat_list, std::vector<nn::NNScore> &outputs);
    static constexpr int GPU_NUM = 1;
private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Lockable lock_gpu;
    int gpu_id = 0;
    bool use_cuda = false;
};

extern GPUModel g_gpu_model[GPUModel::GPU_NUM];

void GPUModel::load_model(const int id) {
    this->gpu_id = id;
    this->lock_gpu.lock();
    Tee<<"load_model("<<id<<")...";
    REP(i, 10) {
        try {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            bool cuda_ok = true;
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, id));
            } catch (const Ort::Exception& e) {
                Tee<<"\nCUDA EP unavailable: "<<e.what()<<" -> falling back to CPU EP\n";
                cuda_ok = false;
            }
            this->session = std::make_unique<Ort::Session>(this->env, "./model/best_single.onnx", session_options);
            this->use_cuda = cuda_ok;
            Tee<<(this->use_cuda ? "GPU" : "CPU")<<" end\n";
            this->lock_gpu.unlock();
            return;
        } catch (const Ort::Exception& e) {
            Tee << "error loading the model: " << e.what() << "\n";
            my_sleep(1000);
            continue;
        }
    }
    this->lock_gpu.unlock();
    std::exit(EXIT_FAILURE);
}

void GPUModel::predict(std::vector<nn::Feature> &feat_list, std::vector<nn::NNScore> &outputs) {
    const auto batch = static_cast<int64_t>(feat_list.size());
    if (batch == 0) {
        return;
    }
    // Flatten features in the same layout as the LibTorch version:
    // per sample, channel-major (FEAT_SIZE) x (FILE_SIZE x RANK_SIZE) values.
    std::vector<float> input_data;
    input_data.reserve(static_cast<size_t>(batch) * nn::FEAT_SIZE * SQUARE_SIZE);
    for (auto &feat : feat_list) {
        REP(c, nn::FEAT_SIZE) {
            REP(s, SQUARE_SIZE) {
                input_data.push_back(static_cast<float>(feat[c][s]));
            }
        }
    }
    const std::array<int64_t, 4> input_shape{batch, nn::FEAT_SIZE, FILE_SIZE, RANK_SIZE};
    const char* input_names[] = {"input"};
    const char* output_names[] = {"value"};

    this->lock_gpu.lock();
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
        auto output_tensors = this->session->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        const float* out = output_tensors[0].GetTensorMutableData<float>();
        REP(i, static_cast<int>(batch)) {
            outputs.push_back(static_cast<double>(out[i]));
        }
    } catch (const Ort::Exception& e) {
        this->lock_gpu.unlock();
        Tee << "onnxruntime predict error: " << e.what() << "\n";
        throw;
    }
    this->lock_gpu.unlock();
}

void predict(const int gpu_id, std::vector<nn::Feature> &feat_list, std::vector<nn::NNScore> &outputs) {
    g_gpu_model[gpu_id].predict(feat_list, outputs);
}

void test_model() {
    g_gpu_model[0].load_model(0);
    game::Position pos;
    pos = hash::hirate();
    auto feat = nn::feature(pos);
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> output_list;
    feat_list.push_back(feat);
    feat_list.push_back(feat);
    feat_list.push_back(feat);
    predict(0, feat_list, output_list);
    Tee<<output_list[0]<<std::endl;
}
nn::NNScore predict_problem(const Key k) {
    auto pos = hash::from_hash(k);
    if (pos.is_win()) {
        return nn::NNScore(0.99);
    } else if (pos.is_lose()) {
        return nn::NNScore(-0.99);
    }
    auto feat = nn::feature(pos);
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> output_list;
    feat_list.push_back(feat);
    predict(0, feat_list, output_list);
    return output_list[0];
}
void test_oracle_model() {
    oracle::g_oracle.load();
    auto correct = 0;
    auto incorrect = 0;
    Tee<<"correct:"<<correct<<" incorrect:"<<incorrect<<"\n";
    Tee<<"correct:"<<double(correct)/double(correct+incorrect)<<"\n";

    REP(i, oracle::OracleData::win_size) {
        const auto k = static_cast<Key>(oracle::g_oracle.win(i));
        const auto score = predict_problem(k);
        if (score < 0.0) {
            incorrect++;
        } else if (score >0.0){
            correct++;
        }
        if(i % 1000 == 0)
            Tee<<"correct:"<<double(correct)/double(correct+incorrect)<<" correct:"<<correct<<" incorrect:"<<incorrect<<"\r";
    }
    Tee<<"correct:"<<correct<<" incorrect:"<<incorrect<<"\n";
    Tee<<"correct:"<<double(correct)/double(correct+incorrect)<<"\n";
}

}

#endif
