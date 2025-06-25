/// \file tokenizer.cpp
/// \brief Tokenizer implementation for text encoding/decoding
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.1.0
#include "tokenizer/tokenizer.hpp"
#include <cstdint>
#include <unordered_map>
#include <regex>

/// \brief Constructor
/// \param model_path the model path
Tokenizer::Tokenizer(const std::string& model_path) {
    std::ifstream fs(model_path + "/tokenizer.json", std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << model_path + "/tokenizer.json" << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    this->tokenizer = tokenizers::Tokenizer::FromBlobJSON(data);
}

/// \brief Destructor
Tokenizer::~Tokenizer() = default;

/// \brief Make the inverse byte map
/// \return the inverse byte map
std::unordered_map<uint32_t, uint8_t> Tokenizer::make_inverse_byte_map() {
    // 1) Build the "printable" byte list bs exactly as in GPT-2/LLaMA
    std::vector<uint8_t> bs;
    for (int b = 33; b <= 126; b++) bs.push_back((uint8_t)b);
    for (int b = 161; b <= 172; b++) bs.push_back((uint8_t)b);
    for (int b = 174; b <= 255; b++) bs.push_back((uint8_t)b);

    // 2) Build the codepoint list cs (same length, but extended to 256)
    std::vector<uint32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), (uint8_t)b) == bs.end()) {
            bs.push_back((uint8_t)b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // 3) Zip them into an inverse map: cp → original byte
    std::unordered_map<uint32_t, uint8_t> inv;
    for (size_t i = 0; i < bs.size(); i++) {
        inv[ cs[i] ] = bs[i];
    }
    return inv;
}

/// \brief Convert the codepoint to utf8
/// \param input the input
/// \return the utf8 string
std::string Tokenizer::cpt_to_utf8(const std::string& input) {
    static auto inv_map = this->make_inverse_byte_map();
    std::string output = "";
    size_t i = 0;

    while (i < input.size()) {
        unsigned char c = input[i];
        uint32_t cp;
        size_t width;

        // --- Decode a UTF-8 codepoint ---
        if (c < 0x80) {
            cp = c;
            width = 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= input.size()) throw std::runtime_error("Truncated UTF-8, ll = 2");
            cp  = (uint32_t)(c     & 0x1F) << 6;
            cp |= (uint32_t)(input[i+1] & 0x3F);
            width = 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= input.size()) throw std::runtime_error("Truncated UTF-8, ll = 3");
            cp  = (uint32_t)(c     & 0x0F) << 12;
            cp |= (uint32_t)(input[i+1] & 0x3F) << 6;
            cp |= (uint32_t)(input[i+2] & 0x3F);
            width = 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= input.size()) throw std::runtime_error("Truncated UTF-8, ll = 4");
            cp  = (uint32_t)(c     & 0x07) << 18;
            cp |= (uint32_t)(input[i+1] & 0x3F) << 12;
            cp |= (uint32_t)(input[i+2] & 0x3F) << 6;
            cp |= (uint32_t)(input[i+3] & 0x3F);
            width = 4;
        }
        else {
            std::cout << std::hex <<"cp: " << cp << std::endl;
            throw std::runtime_error("Invalid UTF-8 byte");
        }

        // --- Look up the original byte ---
        auto it = inv_map.find(cp);
        if (it == inv_map.end()) {
            throw std::runtime_error("Codepoint not in LLaMA byte map");
        }
        output.push_back((char)it->second);

        i += width;
    }

    return output;
}

/// \brief Encode the text
/// \param text the text
/// \return the encoded tokens
std::vector<int> Tokenizer::encode(const std::string& text) {
    return this->tokenizer->Encode(text);
}

/// \brief Decode the tokens
/// \param tokens the tokens
/// \return the decoded text
std::string Tokenizer::decode(const std::vector<int>& tokens) {
    return this->tokenizer->Decode(tokens);
}

/// \brief Run time decoder
/// \param answer_token the answer token
/// \return the decoded text
std::string Tokenizer::run_time_decoder(int answer_token) {
    return this->cpt_to_utf8(this->tokenizer->IdToToken(answer_token));
}
