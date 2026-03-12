/**
 * @file test_status.cpp
 * @brief 测试 base::Status 类的功能
 */

#include <gtest/gtest.h>
#include "base/base.h"
#include <sstream>

using namespace base;
using namespace base::error;

// ==================== Status 构造和基本操作测试 ====================

/**
 * @test 测试默认构造的 Status 应该是成功状态
 */
TEST(StatusTest, DefaultConstructorShouldBeSuccess) {
    Status status;
    EXPECT_TRUE(status);
    EXPECT_TRUE(status == StatusCode::kSuccess);
    EXPECT_EQ(int(status), StatusCode::kSuccess);
    EXPECT_EQ(status.get_err_code(), StatusCode::kSuccess);
}

/**
 * @test 测试带错误码和消息的构造
 */
TEST(StatusTest, ConstructorWithCodeAndMessage) {
    Status status(StatusCode::kInternalError, "Test error message");

    EXPECT_FALSE(status);
    EXPECT_TRUE(status == StatusCode::kInternalError);
    EXPECT_EQ(status.get_err_code(), StatusCode::kInternalError);
    EXPECT_EQ(status.get_err_msg(), "Test error message");
}

/**
 * @test 测试拷贝构造
 */
TEST(StatusTest, CopyConstructor) {
    Status original(StatusCode::kPathNotValid, "Path not found");
    Status copied = original;

    EXPECT_EQ(copied.get_err_code(), original.get_err_code());
    EXPECT_EQ(copied.get_err_msg(), original.get_err_msg());
}

/**
 * @test 测试赋值运算符
 */
TEST(StatusTest, AssignmentOperator) {
    Status original(StatusCode::kModelParseError, "Parse failed");
    Status assigned;
    assigned = original;

    EXPECT_EQ(assigned.get_err_code(), original.get_err_code());
    EXPECT_EQ(assigned.get_err_msg(), original.get_err_msg());
}

/**
 * @test 测试整数赋值
 */
TEST(StatusTest, IntAssignment) {
    Status status;
    status = StatusCode::kInvalidArgument;

    EXPECT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(status);
}

// ==================== Status 比较操作测试 ====================

/**
 * @test 测试相等比较
 */
TEST(StatusTest, EqualityComparison) {
    Status status(StatusCode::kInternalError, "Error");

    EXPECT_TRUE(status == StatusCode::kInternalError);
    EXPECT_FALSE(status == StatusCode::kSuccess);
}

/**
 * @test 测试不等比较
 */
TEST(StatusTest, InequalityComparison) {
    Status status(StatusCode::kInternalError, "Error");

    EXPECT_TRUE(status != StatusCode::kSuccess);
    EXPECT_FALSE(status != StatusCode::kInternalError);
}

/**
 * @test 测试布尔转换（成功为 true）
 */
TEST(StatusTest, BoolConversionSuccess) {
    Status status;
    bool success = status;

    EXPECT_TRUE(success);
}

/**
 * @test 测试布尔转换（失败为 false）
 */
TEST(StatusTest, BoolConversionFailure) {
    Status status(StatusCode::kInternalError, "Error");
    bool success = status;

    EXPECT_FALSE(success);
}

/**
 * @test 测试整数转换
 */
TEST(StatusTest, IntConversion) {
    Status status(StatusCode::kPathNotValid, "Error");
    int code = status;

    EXPECT_EQ(code, StatusCode::kPathNotValid);
}

// ==================== Status 消息操作测试 ====================

/**
 * @test 测试设置错误消息
 */
TEST(StatusTest, SetErrorMessage) {
    Status status;
    status.set_err_msg("New error message");

    EXPECT_EQ(status.get_err_msg(), "New error message");
}

/**
 * @test 测试获取错误消息
 */
TEST(StatusTest, GetErrorMessage) {
    Status status(StatusCode::kInternalError, "Test message");

    EXPECT_EQ(status.get_err_msg(), "Test message");
}

// ==================== error 命名空间函数测试 ====================

/**
 * @test 测试 Success 函数
 */
TEST(StatusTest, SuccessFunction) {
    Status status = Success("Operation completed");

    EXPECT_TRUE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kSuccess);
    EXPECT_EQ(status.get_err_msg(), "Operation completed");
}

/**
 * @test 测试 FunctionNotImplement 函数
 */
TEST(StatusTest, FunctionNotImplementFunction) {
    Status status = FunctionNotImplement("Feature not implemented");

    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);
    EXPECT_EQ(status.get_err_msg(), "Feature not implemented");
}

/**
 * @test 测试 PathNotValid 函数
 */
TEST(StatusTest, PathNotValidFunction) {
    Status status = PathNotValid("/invalid/path");

    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kPathNotValid);
    EXPECT_EQ(status.get_err_msg(), "/invalid/path");
}

/**
 * @test 测试 ModelParseError 函数
 */
TEST(StatusTest, ModelParseErrorFunction) {
    Status status = ModelParseError("Failed to parse model");

    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kModelParseError);
    EXPECT_EQ(status.get_err_msg(), "Failed to parse model");
}

/**
 * @test 测试 InternalError 函数
 */
TEST(StatusTest, InternalErrorFunction) {
    Status status = InternalError("Internal failure");

    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kInternalError);
    EXPECT_EQ(status.get_err_msg(), "Internal failure");
}

/**
 * @test 测试 KeyHasExits 函数
 */
TEST(StatusTest, KeyHasExitsFunction) {
    Status status = KeyHasExits("Key already exists");

    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kKeyValueHasExist);
    EXPECT_EQ(status.get_err_msg(), "Key already exists");
}

/**
 * @test 测试 InvalidArgument 函数
 */
TEST(StatusTest, InvalidArgumentFunction) {
    Status status = InvalidArgument("Invalid argument provided");

    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.get_err_msg(), "Invalid argument provided");
}

// ==================== Status 输出测试 ====================

/**
 * @test 测试流输出运算符
 */
TEST(StatusTest, OutputStreamOperator) {
    Status status(StatusCode::kInternalError, "Test error");
    std::stringstream ss;
    ss << status;

    EXPECT_EQ(ss.str(), "Test error");
}

/**
 * @test 测试空消息的输出
 */
TEST(StatusTest, OutputStreamOperatorWithEmptyMessage) {
    Status status;
    std::stringstream ss;
    ss << status;

    EXPECT_EQ(ss.str(), "");
}

// ==================== Status 链式操作测试 ====================

/**
 * @test 测试多次赋值
 */
TEST(StatusTest, MultipleAssignments) {
    Status status;

    status = StatusCode::kSuccess;
    EXPECT_TRUE(status);

    status = StatusCode::kInternalError;
    EXPECT_FALSE(status);

    status = StatusCode::kSuccess;
    EXPECT_TRUE(status);
}

/**
 * @test 测试状态传递
 */
TEST(StatusTest, StatusPassing) {
    auto func = []() -> Status {
        return Success("Function succeeded");
    };

    Status status = func();
    EXPECT_TRUE(status);
    EXPECT_EQ(status.get_err_msg(), "Function succeeded");
}

// ==================== Status 常量测试 ====================

/**
 * @test 测试所有状态码常量
 */
TEST(StatusTest, AllStatusCodeConstants) {
    EXPECT_EQ(StatusCode::kSuccess, 0);
    EXPECT_EQ(StatusCode::kFunctionUnImplement, 1);
    EXPECT_EQ(StatusCode::kPathNotValid, 2);
    EXPECT_EQ(StatusCode::kModelParseError, 3);
    EXPECT_EQ(StatusCode::kInternalError, 5);
    EXPECT_EQ(StatusCode::kKeyValueHasExist, 6);
    EXPECT_EQ(StatusCode::kInvalidArgument, 7);
}

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
