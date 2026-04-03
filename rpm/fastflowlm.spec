%global flm_version 0.9.38
%global install_prefix /opt/fastflowlm

Name:           fastflowlm
Version:        %{flm_version}
Release:        1%{?dist}
Summary:        FastFlowLM CLI runtime for AMD Ryzen AI NPUs
License:        MIT and Proprietary
URL:            https://github.com/FastFlowLM/FastFlowLM

Source0:        %{name}-%{version}.tar.gz

ExclusiveArch:  x86_64

BuildRequires:  cmake >= 3.22
BuildRequires:  ninja-build
BuildRequires:  gcc-c++
BuildRequires:  git
BuildRequires:  cargo
BuildRequires:  rust
BuildRequires:  pkgconfig
BuildRequires:  boost-devel
BuildRequires:  boost-program-options
BuildRequires:  libcurl-devel
BuildRequires:  fftw-devel
BuildRequires:  readline-devel
BuildRequires:  libuuid-devel
BuildRequires:  libdrm-devel
BuildRequires:  ffmpeg-free-devel

Requires:       xrt

%description
FastFlowLM is an NPU-first runtime for running Large Language Models
on AMD Ryzen AI NPUs. It provides a CLI and an OpenAI-compatible
REST API server.

# Disable debug package generation
%global debug_package %{nil}

# Disable stripping — matches Debian packaging workaround
%global __strip /bin/true

# Disable RPATH check since we set a custom RPATH
%define __arch_install_post %{nil}

%prep
%autosetup -n %{name}-%{version}

%build
cmake -S src -B build --preset linux-default
cmake --build build %{?_smp_mflags}

%install
DESTDIR=%{buildroot} cmake --install build --prefix=%{install_prefix}

# Remove include directories (not needed at runtime)
rm -rf %{buildroot}%{install_prefix}/include

# Create /usr/bin symlink
install -d %{buildroot}%{_bindir}
ln -sf %{install_prefix}/bin/flm %{buildroot}%{_bindir}/flm

# Remove the cmake-generated symlink in /usr/local/bin if present
rm -rf %{buildroot}/usr/local/bin

%files
%license LICENSE_BINARY.txt LICENSE_RUNTIME.txt
%doc README.md
%{install_prefix}/
%{_bindir}/flm

%changelog
* Thu Apr 03 2026 FastFlowLM <noreply@example.com> - 0.9.38-1
- Initial RPM package
