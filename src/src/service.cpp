#include "service.hpp"
#include "utils/utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <pwd.h>
#include <unistd.h>
#include <sys/stat.h>

namespace service_cmd {

static std::string resolve_model_path() {
    const char* sudo_user = std::getenv("SUDO_USER");
    if (sudo_user && strlen(sudo_user) > 0) {
        struct passwd* pw = getpwnam(sudo_user);
        if (pw && pw->pw_dir) {
            return std::string(pw->pw_dir) + "/.config/flm";
        }
    }
    return utils::get_models_directory();
}

static std::string resolve_flm_binary() {
    return utils::get_executable_directory() + "/flm";
}

static bool run_command(const std::string& cmd, bool report_errors = true) {
    int ret = system(cmd.c_str());
    if (ret != 0 && report_errors) {
        std::cerr << "Command failed: " << cmd << std::endl;
    }
    return ret == 0;
}

static bool is_root() {
    return geteuid() == 0;
}

static bool has_systemd() {
    return std::filesystem::exists("/run/systemd/system");
}

static bool file_write(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write to " << path << std::endl;
        return false;
    }
    f << content;
    f.close();
    return true;
}

static int do_install(const program_args_t& args) {
    if (!is_root()) {
        std::cerr << "Error: flm service install must be run as root (use sudo)" << std::endl;
        return 1;
    }
    if (!has_systemd()) {
        std::cerr << "Error: systemd not detected on this system" << std::endl;
        return 1;
    }
    if (std::filesystem::exists("/etc/systemd/system/flm.service")) {
        std::cerr << "Error: flm service is already installed. Run 'flm service uninstall' first." << std::endl;
        return 1;
    }

    // Create flm system user if it doesn't exist
    if (getpwnam("flm") == nullptr) {
        if (!run_command("useradd --system --no-create-home --shell /usr/sbin/nologin flm")) {
            std::cerr << "Error: Failed to create flm system user" << std::endl;
            return 1;
        }
        std::cout << "Created system user 'flm'" << std::endl;
    }

    // Add flm user to render group for NPU access
    run_command("usermod -a -G render flm", false);

    // Resolve paths
    std::string model_path = resolve_model_path();
    std::string flm_binary = resolve_flm_binary();
    int port = utils::get_server_port(args.port);
    std::string host = args.host;

    // Create /etc/flm directory
    std::filesystem::create_directories("/etc/flm");

    // Write config file
    std::ostringstream conf;
    conf << "# FastFlowLM service configuration\n";
    conf << "FLM_HOST=" << host << "\n";
    conf << "FLM_PORT=" << port << "\n";
    conf << "FLM_MODEL_PATH=" << model_path << "\n";
    if (!file_write("/etc/flm/service.conf", conf.str())) return 1;

    // Write wrapper script
    std::ostringstream wrapper;
    wrapper << "#!/bin/bash\n";
    wrapper << "source /etc/flm/service.conf\n";
    wrapper << "export FLM_MODEL_PATH\n";
    wrapper << "exec " << flm_binary << " serve --quiet \\\n";
    wrapper << "    --host \"${FLM_HOST}\" \\\n";
    wrapper << "    --port \"${FLM_PORT}\"\n";
    if (!file_write("/etc/flm/flm-service-wrapper", wrapper.str())) return 1;
    chmod("/etc/flm/flm-service-wrapper", 0755);

    // Write systemd unit
    std::ostringstream unit;
    unit << "[Unit]\n";
    unit << "Description=FastFlowLM Inference Server\n";
    unit << "After=network.target\n";
    unit << "\n";
    unit << "[Service]\n";
    unit << "Type=simple\n";
    unit << "User=flm\n";
    unit << "Group=render\n";
    unit << "ExecStart=/etc/flm/flm-service-wrapper\n";
    unit << "Restart=on-failure\n";
    unit << "RestartSec=5\n";
    unit << "LimitMEMLOCK=infinity\n";
    unit << "NoNewPrivileges=true\n";
    unit << "ProtectSystem=strict\n";
    unit << "ProtectHome=read-only\n";
    unit << "PrivateTmp=true\n";
    unit << "StandardOutput=journal\n";
    unit << "StandardError=journal\n";
    unit << "SyslogIdentifier=flm\n";
    unit << "\n";
    unit << "[Install]\n";
    unit << "WantedBy=multi-user.target\n";
    if (!file_write("/etc/systemd/system/flm.service", unit.str())) return 1;

    // Reload and enable
    run_command("systemctl daemon-reload");
    run_command("systemctl enable flm.service");

    std::cout << "FastFlowLM service installed successfully." << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Host:        " << host << std::endl;
    std::cout << "  Port:        " << port << std::endl;
    std::cout << "  Model path:  " << model_path << std::endl;
    std::cout << "  Binary:      " << flm_binary << std::endl;
    std::cout << std::endl;
    std::cout << "Next steps:" << std::endl;
    std::cout << "  sudo flm service start      - Start the service" << std::endl;
    std::cout << "  flm service status           - Check service status" << std::endl;
    std::cout << "  journalctl -u flm -f         - View logs" << std::endl;

    return 0;
}

static int do_uninstall() {
    if (!is_root()) {
        std::cerr << "Error: flm service uninstall must be run as root (use sudo)" << std::endl;
        return 1;
    }

    run_command("systemctl stop flm.service", false);
    run_command("systemctl disable flm.service", false);

    std::filesystem::remove("/etc/systemd/system/flm.service");
    std::filesystem::remove("/etc/flm/flm-service-wrapper");
    std::filesystem::remove("/etc/flm/service.conf");
    std::filesystem::remove("/etc/flm");

    run_command("systemctl daemon-reload");

    std::cout << "FastFlowLM service uninstalled." << std::endl;
    std::cout << "Note: The 'flm' system user was not removed (may own files)." << std::endl;
    return 0;
}

static int do_start() {
    if (!is_root()) {
        std::cerr << "Error: flm service start must be run as root (use sudo)" << std::endl;
        return 1;
    }
    return run_command("systemctl start flm.service") ? 0 : 1;
}

static int do_stop() {
    if (!is_root()) {
        std::cerr << "Error: flm service stop must be run as root (use sudo)" << std::endl;
        return 1;
    }
    return run_command("systemctl stop flm.service") ? 0 : 1;
}

static int do_status() {
    // status doesn't require root
    return run_command("systemctl status flm.service") ? 0 : 1;
}

int handle_service_command(const std::string& subcommand, const program_args_t& args) {
    if (subcommand == "install")   return do_install(args);
    if (subcommand == "uninstall") return do_uninstall();
    if (subcommand == "start")     return do_start();
    if (subcommand == "stop")      return do_stop();
    if (subcommand == "status")    return do_status();

    std::cerr << "Unknown service action: " << subcommand << std::endl;
    std::cerr << "Usage: flm service <install|uninstall|start|stop|status>" << std::endl;
    return 1;
}

} // namespace service_cmd
