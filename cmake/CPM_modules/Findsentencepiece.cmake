include("/home/nozom1/code/cuda/LLama/KuiperLLama/cmake/cmake/CPM_0.39.0.cmake")
CPMAddPackage("NAME;sentencepiece;GITHUB_REPOSITORY;google/sentencepiece;VERSION;0.2.0;OPTIONS;SPM_USE_BUILTIN_ABSL=OFF")
set(sentencepiece_FOUND TRUE)