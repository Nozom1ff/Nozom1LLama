/**
 * @file test_datatype.cpp
 * @brief 测试 DataType 枚举和相关函数
 */

#include <gtest/gtest.h>
#include "base/base.h"

using namespace base;

// ==================== DataType 枚举测试 ====================

/**
 * @test 测试 DataType 枚举值
 */
TEST(DataTypeTest, EnumValues) {
    EXPECT_EQ(uint8_t(DataType::kUnknown), 0);
    EXPECT_EQ(uint8_t(DataType::kTypeFp32), 1);
    EXPECT_EQ(uint8_t(DataType::kTypeInt8), 2);
    EXPECT_EQ(uint8_t(DataType::kTypeInt32), 3);
    EXPECT_EQ(uint8_t(DataType::kTypeFp16), 4);
}

// ==================== DataTypeSize 函数测试 ====================

/**
 * @test 测试 FP32 数据类型大小
 */
TEST(DataTypeTest, SizeOfFp32) {
    size_t size = DataTypeSize(DataType::kTypeFp32);
    EXPECT_EQ(size, sizeof(float));
    EXPECT_EQ(size, 4);
}

/**
 * @test 测试 FP16 数据类型大小
 */
TEST(DataTypeTest, SizeOfFp16) {
    size_t size = DataTypeSize(DataType::kTypeFp16);
    EXPECT_EQ(size, sizeof(__half));
    EXPECT_EQ(size, 2);
}

/**
 * @test 测试 Int8 数据类型大小
 */
TEST(DataTypeTest, SizeOfInt8) {
    size_t size = DataTypeSize(DataType::kTypeInt8);
    EXPECT_EQ(size, sizeof(int8_t));
    EXPECT_EQ(size, 1);
}

/**
 * @test 测试 Int32 数据类型大小
 */
TEST(DataTypeTest, SizeOfInt32) {
    size_t size = DataTypeSize(DataType::kTypeInt32);
    EXPECT_EQ(size, sizeof(int32_t));
    EXPECT_EQ(size, 4);
}

/**
 * @test 测试 Unknown 数据类型大小
 */
TEST(DataTypeTest, SizeOfUnknown) {
    size_t size = DataTypeSize(DataType::kUnknown);
    EXPECT_EQ(size, 0);
}

// ==================== DeviceType 枚举测试 ====================

/**
 * @test 测试 DeviceType 枚举值
 */
TEST(DeviceTypeTest, EnumValues) {
    EXPECT_EQ(uint8_t(DeviceType::kUnknown), 0);
    EXPECT_EQ(uint8_t(DeviceType::kCPU), 1);
    EXPECT_EQ(uint8_t(DeviceType::kCUDA), 2);
}

/**
 * @test 测试 DeviceType 比较
 */
TEST(DeviceTypeTest, Comparison) {
    DeviceType cpu = DeviceType::kCPU;
    DeviceType cuda = DeviceType::kCUDA;

    EXPECT_TRUE(cpu == DeviceType::kCPU);
    EXPECT_TRUE(cpu != DeviceType::kCUDA);
    EXPECT_TRUE(cuda == DeviceType::kCUDA);
    EXPECT_TRUE(cuda != DeviceType::kCPU);
}

// ==================== ModelType 枚举测试 ====================

/**
 * @test 测试 ModelType 枚举值
 */
TEST(ModelTypeTest, EnumValues) {
    EXPECT_EQ(uint8_t(ModelType::kUnknown), 0);
    EXPECT_EQ(uint8_t(ModelType::kLLama2), 1);
}

// ==================== ModelBufferType 枚举测试 ====================

/**
 * @test 测试 ModelBufferType 枚举值
 */
TEST(ModelBufferTypeTest, EnumValues) {
    using namespace model;

    EXPECT_EQ(int(ModelBufferType::kInputTokens), 0);
    EXPECT_EQ(int(ModelBufferType::kInputEmbeddings), 1);
    EXPECT_EQ(int(ModelBufferType::kOutputRMSNorm), 2);
    EXPECT_EQ(int(ModelBufferType::kKeyCache), 3);
    EXPECT_EQ(int(ModelBufferType::kValueCache), 4);
    EXPECT_EQ(int(ModelBufferType::kQuery), 5);
    EXPECT_EQ(int(ModelBufferType::kInputPos), 6);
    EXPECT_EQ(int(ModelBufferType::kScoreStorage), 7);
    EXPECT_EQ(int(ModelBufferType::kOutputMHA), 8);
    EXPECT_EQ(int(ModelBufferType::kAttnOutput), 9);
    EXPECT_EQ(int(ModelBufferType::kW1Output), 10);
    EXPECT_EQ(int(ModelBufferType::kW2Output), 11);
    EXPECT_EQ(int(ModelBufferType::kW3Output), 12);
    EXPECT_EQ(int(ModelBufferType::kFFNRMSNorm), 13);
    EXPECT_EQ(int(ModelBufferType::kForwardOutput), 15);
    EXPECT_EQ(int(ModelBufferType::kForwardOutputCPU), 16);
    EXPECT_EQ(int(ModelBufferType::kSinCache), 17);
    EXPECT_EQ(int(ModelBufferType::kCosCache), 18);
}

// ==================== TokenizerType 枚举测试 ====================

/**
 * @test 测试 TokenizerType 枚举值
 */
TEST(TokenizerTypeTest, EnumValues) {
    EXPECT_EQ(int(TokenizerType::kEncodeUnknown), -1);
    EXPECT_EQ(int(TokenizerType::kEncodeSpe), 0);
    EXPECT_EQ(int(TokenizerType::kEncodeBpe), 1);
}

// ==================== NoCopyable 类测试 ====================

/**
 * @test 测试 NoCopyable 防止拷贝
 */
TEST(NoCopyableTest, PreventCopy) {
    class TestClass : public base::NoCopyable {
    public:
        TestClass() = default;
        int value = 42;
    };

    TestClass obj1;
    TestClass obj2;

    // 不应该编译（如果取消注释会编译失败）
    // obj1 = obj2;  // 编译错误：删除的拷贝赋值运算符
    // TestClass obj3 = obj1;  // 编译错误：删除的拷贝构造函数

    // 应该能正常移动
    EXPECT_EQ(obj1.value, 42);
    EXPECT_EQ(obj2.value, 42);
}

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
