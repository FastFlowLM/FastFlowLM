#pragma once
#include <string>
#include "program_args.hpp"

namespace service_cmd {
    int handle_service_command(const std::string& subcommand, const program_args_t& args);
}
