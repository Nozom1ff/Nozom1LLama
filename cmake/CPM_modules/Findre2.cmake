include("/home/nozom1/code/cuda/LLama/KuiperLLama/cmake/cmake/CPM_0.39.0.cmake")
CPMAddPackage("NAME;re2;GITHUB_REPOSITORY;google/re2;GIT_TAG;2023-07-01;OPTIONS;RE2_BUILD_TESTING OFF")
set(re2_FOUND TRUE)