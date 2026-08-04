add_executable(openai_tool_policy_test test.cpp)
target_compile_features(openai_tool_policy_test PRIVATE cxx_std_20)
target_include_directories(openai_tool_policy_test PRIVATE ../../include)

add_test(NAME openai_tool_policy_contract COMMAND openai_tool_policy_test)
