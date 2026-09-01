#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

inline void Check(
    bool condition,
    std::string_view expression,
    const char* file,
    int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " CHECK failed: " + std::string(expression));
    }
}

#define CHECK(expression) \
    Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

template <class Function>
void CheckThrowsContains(Function&& function, std::string_view expected) {
    try {
        function();
    } catch (const std::exception& error) {
        CHECK(std::string_view(error.what()).find(expected) !=
              std::string_view::npos);
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}
