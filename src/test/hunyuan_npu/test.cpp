/// \file test.cpp
/// \brief standalone benchmark harness for the hunyuan-dense engine (Hy-MT2-1.8B)
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \note The model is a translator, so every turn is independent: the context is
///       cleared before each prompt and the history never accumulates. Each
///       prompt is the fixed translation instruction followed by the source text.
///       The harness walks the whole danmaku corpus in bench_samples.hpp, throws
///       away the first few turns as warm-up, and reports the distribution of the
///       end-to-end latency over the rest.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#define NOMINMAX
#ifdef __WINDOWS__
#include <windows.h>
#endif
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hunyuan.hpp"
#include "model_list.hpp"
#include "bench_samples.hpp"

flm_rt::device npu_device_global;

/// \brief the fixed translator instruction every prompt starts with
static const std::string kTranslatePrefix =
    "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释。"
    "输出必须全部使用英语，不要输出源语言或原文：";

/// \brief turns discarded before the timings are collected
/// \note the first calls pay for the lazy xclbin / weight paging, so they sit an
///       order of magnitude above steady state and would swamp every statistic.
static const int kDefaultWarmup = 3;

/// \brief one measured translation turn
struct turn_result_t {
    std::string source;       ///< the danmaku line that was translated
    std::string translation;  ///< what the model produced, trimmed
    double prefill_ms = 0.0;  ///< wall clock spent inserting the prompt (ttft floor)
    double decode_ms = 0.0;   ///< wall clock spent generating
    double total_ms = 0.0;    ///< end to end, clear_context through last token
    int prompt_tokens = 0;    ///< tokens after the chat template, instruction included
    int gen_tokens = 0;       ///< tokens the model emitted
};

/// \brief milliseconds between two steady-clock stamps
static double ms_since(const std::chrono::steady_clock::time_point& t0) {
    std::chrono::duration<double, std::milli> d = std::chrono::steady_clock::now() - t0;
    return d.count();
}

/// \brief strip leading / trailing whitespace so empty answers are detectable
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

/// \brief escape a field for CSV: wrap in quotes and double any quote inside
static std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        }
        else if (c == '\n' || c == '\r') {
            out += ' ';
        }
        else {
            out += c;
        }
    }
    out += '"';
    return out;
}

/// \brief clear the context and run one standalone translation turn
/// \param chat the loaded engine
/// \param text source text appended to the translator instruction
/// \param length_limit max tokens to generate
/// \return the timings and the output of the turn
/// \note prefill and decode are driven separately so the two phases can be timed
///       apart; the generated text is captured into a buffer instead of being
///       streamed, keeping console I/O out of the measurement.
static turn_result_t run_turn(std::unique_ptr<AutoModel>& chat, const std::string& text,
                              int length_limit) {
    turn_result_t r;
    r.source = text;

    chat->clear_context();

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    messages.push_back({ {"role", "user"}, {"content", kTranslatePrefix + text} });

    chat_meta_info_t meta_info;
    meta_info.restore_allowed = false;

    lm_uniform_input_t input;
    input.messages = messages;

    std::ostringstream sink;

    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    chat->start_total_timer();
    if (!chat->insert(meta_info, input)) {
        chat->stop_total_timer();
        r.total_ms = ms_since(t_start);
        return r;  // prompt rejected: left empty so the caller can flag it
    }
    r.prefill_ms = ms_since(t_start);
    r.prompt_tokens = chat->get_current_context_length();

    std::chrono::steady_clock::time_point t_decode = std::chrono::steady_clock::now();
    std::string out = chat->generate(meta_info, length_limit, sink);
    chat->stop_total_timer();
    r.decode_ms = ms_since(t_decode);
    r.total_ms = ms_since(t_start);
    r.gen_tokens = chat->get_current_context_length() - r.prompt_tokens;
    r.translation = trim(out.empty() ? sink.str() : out);
    return r;
}

/// \brief the distribution of one metric over the measured turns
struct stats_t {
    size_t n = 0;
    double mean = 0.0;
    double stddev = 0.0;  ///< sample standard deviation
    double cv = 0.0;      ///< coefficient of variation, stddev / mean
    double min = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

/// \brief nearest-rank percentile of an already sorted sample
static double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    size_t idx = static_cast<size_t>(std::ceil(q * sorted.size()));
    if (idx == 0) {
        idx = 1;
    }
    if (idx > sorted.size()) {
        idx = sorted.size();
    }
    return sorted[idx - 1];
}

/// \brief summarize a sample
static stats_t summarize(std::vector<double> v) {
    stats_t s;
    s.n = v.size();
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    s.mean = sum / v.size();
    if (v.size() > 1) {
        double acc = 0.0;
        for (double x : v) {
            acc += (x - s.mean) * (x - s.mean);
        }
        s.stddev = std::sqrt(acc / (v.size() - 1));
    }
    s.cv = s.mean > 0.0 ? s.stddev / s.mean : 0.0;
    s.min = v.front();
    s.max = v.back();
    s.p50 = percentile(v, 0.50);
    s.p90 = percentile(v, 0.90);
    s.p95 = percentile(v, 0.95);
    s.p99 = percentile(v, 0.99);
    return s;
}

/// \brief print one summarize() result as a table row
static void print_stats_row(const std::string& name, const stats_t& s, const std::string& unit) {
    std::cout << "  " << std::left << std::setw(22) << (name + " (" + unit + ")")
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << s.mean
              << std::setw(10) << s.stddev
              << std::setw(9) << (s.cv * 100.0)
              << std::setw(10) << s.min
              << std::setw(10) << s.p50
              << std::setw(10) << s.p90
              << std::setw(10) << s.p95
              << std::setw(10) << s.p99
              << std::setw(10) << s.max
              << std::endl;
}

int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif

    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hunyuan:1.8b"), "Model tag");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Length,l", arg_utils::po::value<int>()->default_value(512), "Max generated tokens");
    desc.add_options()("warmup,w", arg_utils::po::value<int>()->default_value(kDefaultWarmup), "Warm-up turns excluded from the statistics");
    desc.add_options()("count,n", arg_utils::po::value<int>()->default_value(0), "Turns to run, 0 = the whole corpus");
    desc.add_options()("csv,c", arg_utils::po::value<std::string>()->default_value("bench.csv"), "Per-turn CSV to write, empty to skip");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool preemption = vm["Preemption"].as<bool>();
    int length_limit = vm["Length"].as<int>();
    int warmup = vm["warmup"].as<int>();
    int count = vm["count"].as<int>();
    std::string csv_path = vm["csv"].as<std::string>();
    std::cout << "Model: " << tag << std::endl;

    size_t total_turns = kBenchSamples.size();
    if (count > 0 && static_cast<size_t>(count) < total_turns) {
        total_turns = static_cast<size_t>(count);
    }
    if (warmup < 0) {
        warmup = 0;
    }
    if (static_cast<size_t>(warmup) >= total_turns) {
        std::cout << "Warm-up (" << warmup << ") covers every one of the " << total_turns
                  << " turns, nothing left to measure" << std::endl;
        return 1;
    }

    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);

    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    std::pair<std::string, nlohmann::json> model_info_pair = model_list.get_model_info(tag);
    nlohmann::json model_info = model_info_pair.second;
    std::cout << "Model path: " << model_path << std::endl;

    if (model_info["details"]["family"] != "hunyuan") {
        std::cout << "Tag '" << tag << "' is not a hunyuan-dense model" << std::endl;
        return 1;
    }

    std::unique_ptr<AutoModel> chat = std::make_unique<Hunyuan>(&npu_device_global);

    // the device must come up after the engine object exists
    npu_device_global = flm_rt::device(0);

    chat->load_model(model_path, model_info, -1, preemption);
    std::cout << chat->show_model_info() << std::endl;

    chat->set_topk(1);   // greedy: deterministic output for testing

    // the instruction itself is constant, so report only its cost in tokens
    std::string prefix = kTranslatePrefix;
    std::cout << "Constant instruction: " << chat->prepare_benchmark(prefix).first
              << " tokens" << std::endl;
    std::cout << "Corpus: " << total_turns << " turns (" << warmup << " warm-up, "
              << (total_turns - warmup) << " measured), length limit " << length_limit
              << std::endl << std::endl;

    std::vector<turn_result_t> results;
    results.reserve(total_turns);
    for (size_t i = 0; i < total_turns; i++) {
        bool is_warmup = i < static_cast<size_t>(warmup);
        turn_result_t r = run_turn(chat, kBenchSamples[i], length_limit);
        results.push_back(r);

        std::cout << "[" << std::setw(3) << (i + 1) << "/" << total_turns << "]"
                  << (is_warmup ? " (warm-up)" : "          ")
                  << " " << std::fixed << std::setprecision(1) << std::setw(8) << r.total_ms
                  << " ms  " << std::setw(3) << r.gen_tokens << " tok  "
                  << r.source << "  ->  "
                  << (r.translation.empty() ? "<EMPTY>" : r.translation) << std::endl;
    }
    std::cout << std::endl;

    // the warm-up turns stay out of every sample below
    std::vector<double> total_ms, prefill_ms, decode_ms, decode_tps, e2e_tps, gen_tokens;
    size_t empty_outputs = 0;
    for (size_t i = static_cast<size_t>(warmup); i < results.size(); i++) {
        const turn_result_t& r = results[i];
        if (r.translation.empty()) {
            empty_outputs++;
        }
        total_ms.push_back(r.total_ms);
        prefill_ms.push_back(r.prefill_ms);
        decode_ms.push_back(r.decode_ms);
        gen_tokens.push_back(static_cast<double>(r.gen_tokens));
        if (r.gen_tokens > 0 && r.decode_ms > 0.0) {
            decode_tps.push_back(r.gen_tokens * 1000.0 / r.decode_ms);
        }
        if (r.gen_tokens > 0 && r.total_ms > 0.0) {
            e2e_tps.push_back(r.gen_tokens * 1000.0 / r.total_ms);
        }
    }

    stats_t total_stats = summarize(total_ms);
    std::cout << "=== Statistics over " << total_stats.n << " measured turns ("
              << warmup << " warm-up turns discarded) ===" << std::endl;
    std::cout << "  " << std::left << std::setw(22) << "metric" << std::right
              << std::setw(10) << "mean" << std::setw(10) << "stddev" << std::setw(9) << "cv%"
              << std::setw(10) << "min" << std::setw(10) << "p50" << std::setw(10) << "p90"
              << std::setw(10) << "p95" << std::setw(10) << "p99" << std::setw(10) << "max"
              << std::endl;
    print_stats_row("end-to-end", total_stats, "ms");
    print_stats_row("prefill", summarize(prefill_ms), "ms");
    print_stats_row("decode", summarize(decode_ms), "ms");
    print_stats_row("generated", summarize(gen_tokens), "tok");
    print_stats_row("decode speed", summarize(decode_tps), "tok/s");
    print_stats_row("e2e speed", summarize(e2e_tps), "tok/s");

    double wall_ms = 0.0;
    double tokens = 0.0;
    for (size_t i = static_cast<size_t>(warmup); i < results.size(); i++) {
        wall_ms += results[i].total_ms;
        tokens += results[i].gen_tokens;
    }
    std::cout << std::endl
              << "  Measured wall clock:   " << std::fixed << std::setprecision(2)
              << (wall_ms / 1000.0) << " s" << std::endl
              << "  Tokens generated:      " << static_cast<long long>(tokens) << std::endl
              << "  Aggregate throughput:  "
              << (wall_ms > 0.0 ? tokens * 1000.0 / wall_ms : 0.0) << " tok/s" << std::endl
              << "  Empty outputs:         " << empty_outputs << " / " << total_stats.n
              << std::endl;

    if (!csv_path.empty()) {
        // binary mode keeps the BOM and the LF line ends byte-exact on Windows
        std::ofstream csv(csv_path, std::ios::binary);
        if (csv.is_open()) {
            // Excel decodes a BOM-less CSV in the system ANSI codepage and mangles
            // the Chinese columns, so the UTF-8 BOM is written up front
            csv << "\xEF\xBB\xBF";
            csv << "index,warmup,prompt_tokens,gen_tokens,prefill_ms,decode_ms,e2e_ms,e2e_tok_per_s,source,translation\n";
            for (size_t i = 0; i < results.size(); i++) {
                const turn_result_t& r = results[i];
                csv << i << "," << (i < static_cast<size_t>(warmup) ? 1 : 0) << ","
                    << r.prompt_tokens << "," << r.gen_tokens << ","
                    << std::fixed << std::setprecision(3)
                    << r.prefill_ms << "," << r.decode_ms << "," << r.total_ms << ","
                    << (r.total_ms > 0.0 ? r.gen_tokens * 1000.0 / r.total_ms : 0.0) << ","
                    << csv_quote(r.source) << "," << csv_quote(r.translation) << "\n";
            }
            std::cout << "  Per-turn CSV:          " << csv_path << std::endl;
        }
        else {
            std::cout << "  Could not open CSV '" << csv_path << "' for writing" << std::endl;
        }
    }

    std::cout << std::endl << chat->show_profile() << std::endl;
    return 0;
}
