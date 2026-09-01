#pragma once

#include <csignal>
#include <stdexcept>
#include <utility>

class ScopedSignalHandler final {
public:
    using Handler = void (*)(int);

    ScopedSignalHandler(int signal_number, Handler handler)
        : signal_number_(signal_number),
          previous_(std::signal(signal_number, handler)) {
        if (previous_ == SIG_ERR) {
            throw std::runtime_error("failed to register signal handler");
        }
    }

    ~ScopedSignalHandler() {
        if (previous_ != SIG_ERR) {
            (void)std::signal(signal_number_, previous_);
        }
    }

    ScopedSignalHandler(const ScopedSignalHandler&) = delete;
    ScopedSignalHandler& operator=(const ScopedSignalHandler&) = delete;

private:
    int signal_number_;
    Handler previous_;
};

template <
    typename StopAndWait,
    typename DestroyServer,
    typename ShutdownRuntime>
bool CompleteServeShutdown(
    StopAndWait&& stop_and_wait,
    DestroyServer&& destroy_server,
    ShutdownRuntime&& shutdown_runtime) {
    std::forward<StopAndWait>(stop_and_wait)();
    std::forward<DestroyServer>(destroy_server)();
    return std::forward<ShutdownRuntime>(shutdown_runtime)();
}
