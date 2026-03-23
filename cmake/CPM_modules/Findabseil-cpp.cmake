include("/home/nozom1/code/cuda/LLama/KuiperLLama/cmake/cmake/CPM_0.39.0.cmake")
CPMAddPackage("NAME;abseil-cpp;GITHUB_REPOSITORY;abseil/abseil-cpp;GIT_TAG;20230802.1;OPTIONS;ABSL_ENABLE_INSTALL ON;ABSL_PROPAGATE_CXX_STD ON")
set(abseil-cpp_FOUND TRUE)