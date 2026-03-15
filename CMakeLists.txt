cmake_minimum_required(VERSION 3.16)
project(nozom1_llama CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 架构（支持 Blackwell 架构 RTX 5090）
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)  # 从 Turing 到 Blackwell (CUDA 13.1+)
set(CMAKE_CUDA_ARCHITECTURES_ALL OFF)

# 导出编译命令数据库
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 CUDA 编译器路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE PATH "CUDA compiler path")

# 查找 CUDA
find_package(CUDA REQUIRED)

# 设置 CUDA 库路径
if(CUDA_FOUND)
    set(CUDA_LIBRARIES_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    if(NOT EXISTS "${CUDA_LIBRARIES_DIR}")
        set(CUDA_LIBRARIES_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
    endif()
    message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES_DIR}")
else()
    message(FATAL_ERROR "CUDA not found!")
endif()

# 选项
option(USE_CPM "Use CPM for dependency management" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# ==================== CPM 依赖管理 ====================
if(USE_CPM)
    # 下载 CPM.cmake
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
        message(STATUS "Downloading CPM.cmake...")
        file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v0.40.2/CPM.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake
        )
    endif()

    include(cmake/CPM.cmake)

    # GTest - 单元测试框架
    CPMAddPackage(
        NAME GTest
        GITHUB_REPOSITORY google/googletest
        VERSION 1.15.0
        OPTIONS "INSTALL_GTEST OFF"
    )

    # GLog - 日志库
    CPMAddPackage(
        NAME glog
        GITHUB_REPOSITORY google/glog
        VERSION 0.7.1
        OPTIONS "BUILD_TESTING OFF"
    )
else()
    # 使用系统安装的依赖
    find_package(GTest REQUIRED)
    find_package(glog REQUIRED)
endif()

# ==================== SentencePiece ====================
find_package(PkgConfig REQUIRED)
pkg_check_modules(SentencePiece REQUIRED sentencepiece)

# ==================== 收集源文件 ====================

# 收集 base 模块源文件
file(GLOB_RECURSE BASE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source/base/*.cpp
)

# 收集 tensor 模块源文件（如果存在）
file(GLOB_RECURSE TENSOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source/tensor/*.cpp
)

# 收集其他模块源文件（如果存在）
file(GLOB_RECURSE OP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source/op/*.cpp
)

file(GLOB_RECURSE MODEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source/model/*.cpp
)

file(GLOB_RECURSE SAMPLER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source/sampler/*.cpp
)

# CUDA 源文件
file(GLOB_RECURSE CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source/op/kernels/cuda/*.cu
)

# ==================== 创建核心库 ====================

# 创建静态库
add_library(nozom1_base STATIC
    ${BASE_SOURCES}
    ${TENSOR_SOURCES}
)

target_include_directories(nozom1_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/include
        ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source
        ${SentencePiece_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(nozom1_base
    PUBLIC
        glog::glog
        ${SentencePiece_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_cuda_rt_LIBRARY}
)

# 添加 CUDA 库搜索路径
target_link_directories(nozom1_base
    PUBLIC
        ${CUDA_LIBRARIES_DIR}
)

# 设置编译选项
target_compile_options(nozom1_base PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
)

# 如果有其他模块源文件，创建完整库
if(TENSOR_SOURCES OR OP_SOURCES OR MODEL_SOURCES)
    add_library(nozom1_core STATIC
        ${BASE_SOURCES}
        ${TENSOR_SOURCES}
        ${OP_SOURCES}
        ${MODEL_SOURCES}
        ${SAMPLER_SOURCES}
        ${CUDA_SOURCES}
    )

    target_include_directories(nozom1_core
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/include
            ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/source
            ${SentencePiece_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(nozom1_core
        PUBLIC
            glog::glog
            ${SentencePiece_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${CUDA_cuda_rt_LIBRARY}
    )

    # 添加 CUDA 库搜索路径
    target_link_directories(nozom1_core
        PUBLIC
            ${CUDA_LIBRARIES_DIR}
    )

    # CUDA 分离编译
    set_target_properties(nozom1_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ==================== 测试 ====================
if(BUILD_TESTS)
    enable_testing()

    # 设置测试输出目录
    set(TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test)

    # 收集测试源文件
    file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    )

    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        get_filename_component(test_dir ${test_source} DIRECTORY)
        get_filename_component(test_category ${test_dir} NAME)

        # 创建测试可执行文件
        add_executable(${test_name} ${test_source})

        # 设置输出目录
        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}/${test_category}
        )

        # 链接库
        if(TARGET nozom1_core)
            target_link_libraries(${test_name}
                PRIVATE
                    nozom1_core
                    GTest::gtest
                    GTest::gtest_main
                    glog::glog
                    ${CUDA_LIBRARIES}
                    ${CUDA_cuda_rt_LIBRARY}
            )
        else()
            target_link_libraries(${test_name}
                PRIVATE
                    nozom1_base
                    GTest::gtest
                    GTest::gtest_main
                    glog::glog
                    ${CUDA_LIBRARIES}
                    ${CUDA_cuda_rt_LIBRARY}
            )
        endif()

        target_include_directories(${test_name}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/nozom1/include
        )

        # 添加到 CTest
        add_test(NAME ${test_category}_${test_name} COMMAND ${test_name})
        # 设置工作目录
        set_tests_properties(${test_category}_${test_name} PROPERTIES
            WORKING_DIRECTORY ${TEST_OUTPUT_DIR}/${test_category}
        )
    endforeach()
endif()

# ==================== 安装 ====================

# 安装头文件
install(DIRECTORY nozom1/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装库
install(TARGETS nozom1_base
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# ==================== 总结 ====================
message(STATUS "")
message(STATUS "==================== Nozom1LLama Configuration ====================")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard:        ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Build tests:          ${BUILD_TESTS}")
message(STATUS "  Build benchmarks:     ${BUILD_BENCHMARKS}")
message(STATUS "  Use CPM:              ${USE_CPM}")
message(STATUS "====================================================================")
message(STATUS "")
