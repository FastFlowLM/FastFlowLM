#pragma once

#include <ryzenai/corelib.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace flm::test {

std::unordered_map<std::string, void*> CompleteCorelibResolver();

void ResetFakeCorelib();
void SetLastErrorMessage(std::string message);
// Drives the API-5 load-time version gate.
void SetFakeCorelibVersion(
    std::uint32_t major,
    std::uint32_t minor,
    std::uint32_t patch) noexcept;
void SetSelftestStatus(ryzenai_corelib_status status) noexcept;
void SetHasDeviceContext(bool value) noexcept;
std::size_t ObjectReleaseCount() noexcept;
std::size_t ObjectReleaseCountFor(void* value) noexcept;
void* LastReleasedObject() noexcept;
std::size_t CleanupCount() noexcept;
std::vector<std::string> FakeCorelibEvents();

}  // namespace flm::test
