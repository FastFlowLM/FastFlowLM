# FastFlowLM Release

Public version.

## Installation & Deployment

### Prerequisites
This application requires the **Microsoft Visual C++ Redistributable for Visual Studio 2015-2022** to run on target computers.

### Quick Installation
1. **Download Visual C++ Redistributable:**
   - x64: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - x86: https://aka.ms/vs/17/release/vc_redist.x86.exe

2. **Install as Administrator:**
   - Run the downloaded installer as Administrator
   - Restart computer if prompted

3. **Alternative installation methods:**
   ```powershell
   # Using Windows Package Manager
   winget install Microsoft.VCRedist.2015+.x64
   
   # Using Chocolatey
   choco install vcredist140
   ```


### Building with Static Linking (Recommended)
To avoid the MSVCP140.dll dependency entirely, build with static linking:

```bash
# Download submodule (tokenizer)
git submodule update --init --recursive

# Clean previous build
rm -rf build/

# Configure with static linking
cmake -B build -S . -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded

# Build
cmake --build build --config Release
```

**Comprehensive Static Linking (New):**
The CMakeLists.txt now includes comprehensive static linking that attempts to statically link:
- Visual C++ Runtime libraries
- Windows system libraries (kernel32, user32, etc.)
- Network libraries (ws2_32, wininet, etc.)
- Cryptography libraries (crypt32, bcrypt, etc.)

```bash
# Build with comprehensive static linking
cmake -B build -S .
cmake --build build --config Release

# Check remaining DLL dependencies
cmake --build build --target check_dependencies
```

**Note:** Some custom libraries (XRT, NPU libraries) may still require DLLs if static versions aren't available.

### Portable Build (Self-Contained Binary)

FastFlowLM can be built as a portable distribution with XRT bundled and FFmpeg statically linked. This eliminates the need for system dependencies and creates a self-contained executable.

**Simple portable build:**

```bash
# Use the linux-portable preset
cmake --preset linux-portable
cmake --build build -j$(nproc)
# The portable layout (lib/ bundling + wrapper) is produced at install time
DESTDIR=$PWD/build/install cmake --install build
```

This will:
1. Check if XRT is installed via pkg-config
2. If not found, automatically fetch XRT (v2.21.75) from source and build it
3. Build FFmpeg (v7.1) and zlib as static libraries
4. On install, bundle the shared libraries into `lib/` with `$ORIGIN` RPATH (requires `patchelf`)

> **Note:** The `cmake --install` step is what bundles the shared libraries into `lib/` and installs the wrapper. Without it you get a plain build, not the self-contained distribution.

**What gets statically linked:**
- ✅ FFmpeg (libavformat, libavcodec, libavutil, libswscale, libswresample)
- ✅ zlib

**What gets bundled (dynamic, in `lib/`):**
- XRT (Xilinx Runtime)
- FFTW (libfftw3)
- XDNA driver plugin (`libxrt_driver_xdna.so.2`) — **only if present on the build machine**; it is not built by this preset and must be provided by your system (e.g. `libxrt-npu2`)

**What remains dynamic (from the target system):**
- Model-specific libraries (llama_npu, qwen_npu, etc.)
- System libraries (libc, libm, etc.)

**Manual options:**

```bash
# Enable portable build manually
cmake --preset linux-default -DFLM_PORTABLE_BUILD=ON
cmake --build build -j$(nproc)
DESTDIR=$PWD/build/install cmake --install build
```

**Customizing source versions:**

```bash
cmake --preset linux-portable \
  -DXRT_GIT_TAG=2.21.75 \
  -DFFMPEG_GIT_TAG=n7.1
cmake --build build -j$(nproc)
```

**Benefits:**
- ✅ Maximum portability - fewer system dependencies
- ✅ No system XRT or FFmpeg installation required
- ✅ Reproducible builds with pinned versions
- ✅ Simpler deployment
- ✅ Works across different Linux distributions

**Notes:**
- First build takes longer (~10-15 minutes) as dependencies are compiled
- Subsequent builds are much faster (dependencies are cached)
- Binary size increases by ~20MB due to embedded libraries
- Requires build tools (git, gcc, cmake, make) during build
- When static build is disabled, uses system packages

**XDNA Driver Plugin:**
The XDNA userspace plugin (`libxrt_driver_xdna.so.2`) is a runtime plugin that XRT loads dynamically. It is NOT statically linked. You need to either:
- Install from system packages: `sudo apt install libxrt-npu2` (Ubuntu/Debian)
- Have the plugin in `/usr/lib/` or alongside the binary in `lib/`
- XRT will automatically discover and load the plugin at runtime

### Creating Deployment Package
Use the provided deployment script:

```powershell
# Create deployment package
.\deploy.ps1

# Or specify custom directories
.\deploy.ps1 -BuildDir "build" -OutputDir "deploy"
```

The deployment package will include:
- `flm.exe` - Main executable
- All required DLLs from `lib/` directory
- `model_list.json` - Model configuration
- `INSTALLATION.md` - Installation instructions
- `run_flm.bat` - Easy execution script

### Troubleshooting

**"MSVCP140.dll not found" error:**
1. Install the Visual C++ Redistributable (see Prerequisites above)
2. Ensure you're using the correct architecture (x64/x86)
3. Try running as Administrator
4. Check if antivirus is blocking DLL files

**Other common issues:**
- If you get "VCRUNTIME140.dll not found", install the same Visual C++ Redistributable
- For "libcurl.dll not found", ensure all DLLs from the `lib/` directory are present
- For AMD XDNA/GPU related errors, ensure proper drivers are installed

**Finding Static Library Alternatives:**
Use the provided script to identify static library alternatives:
```powershell
.\find_static_libs.ps1
```

This will help you find static versions of your custom libraries to further reduce DLL dependencies.

### Development
```bash
# Build for development
cmake -B build -S .
cmake --build build --config Debug

# Run tests (if available)
ctest --test-dir build
```
