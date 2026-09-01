#pragma once

#include <ryzenai/corelib.h>

#include <cstddef>
#include <string>
#include <unordered_map>

namespace flm::test {

std::unordered_map<std::string, void*> CompleteCorelibResolver();

void ResetFakeCorelib();
void SetLastErrorMessage(std::string message);
std::size_t ObjectReleaseCount() noexcept;
std::size_t ObjectReleaseCountFor(void* value) noexcept;
void* LastReleasedObject() noexcept;

}  // namespace flm::test
