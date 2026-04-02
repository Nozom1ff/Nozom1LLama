# KuiperLLama Tensor系统图

## 1. Tensor类结构

```mermaid
classDiagram
    class Tensor {
        +int32_t dims_[4]
        +int32_t size_
        +bool is_external_
        +DataType data_type_
        +std::shared_ptr~Buffer~ buffer_
        +DeviceType device_type_

        +Tensor()
        +Tensor(DeviceType, DataType, int32_t, ...)
        +reshape()
        +index~T~(int32_t)
        +ptr~T~()
        +to_cpu()
        +to_cuda()
        +assign()
        +float* ptr()
        +size() int32_t
        +get_dim(int32_t) int32_t
        +is_empty() bool
    }

    class Buffer {
        +std::shared_ptr~DeviceAllocator~ allocator_
        +void* ptr_
        +size_t byte_size_
        +bool use_external_

        +Buffer()
        +Buffer(DeviceAllocator*, size_t, void*)
        +~Buffer()
        +copy_from()
        +ptr~T~() T*
    }

    class DeviceAllocator {
        <<abstract>>
        +allocate(size_t) void*
        +deallocate(void*)
        +memcpy()
        +memset()
    }

    class CpuDeviceAllocator {
        +allocate() void*
        +deallocate()
        +memcpy()
    }

    class CudaDeviceAllocator {
        +allocate() void*
        +deallocate()
        +memcpy()
    }

    Tensor --> Buffer : contains
    Buffer --> DeviceAllocator : uses
    DeviceAllocator <|-- CpuDeviceAllocator
    DeviceAllocator <|-- CudaDeviceAllocator
```

## 2. Tensor内存管理

```mermaid
graph TB
    subgraph Tensor创建
        CREATE["Tensor构造"]
        ALLOC_REQ["请求分配内存"]
        ALLOCATOR["DeviceAllocator"]
        BUFFER["创建Buffer"]
    end

    subgraph 内存来源
        INTERNAL["内部分配<br/>allocator_->allocate()"]
        EXTERNAL["外部内存<br/>use_external_=true"]
    end

    subgraph 设备类型
        CPU_MEM["CPU Memory<br/>malloc"]
        CUDA_MEM["CUDA Memory<br/>cudaMalloc"]
    end

    subgraph 生命周期
        REF_COUNT["引用计数管理"]
        AUTO_FREE["自动释放"]
    end

    CREATE --> ALLOC_REQ
    ALLOC_REQ --> ALLOCATOR
    ALLOCATOR --> BUFFER

    ALLOCATOR --> INTERNAL
    ALLOCATOR --> EXTERNAL

    INTERNAL --> CPU_MEM
    INTERNAL --> CUDA_MEM

    BUFFER --> REF_COUNT
    REF_COUNT --> AUTO_FREE
```

## 3. Tensor维度操作

```mermaid
graph LR
    subgraph 维度定义
        D0["dims_[0]: 第一维"]
        D1["dims_[1]: 第二维"]
        D2["dims_[2]: 第三维"]
        D3["dims_[3]: 第四维"]
    end

    subgraph 常见形状
        EMB["Embedding: [seq_len, dim]"]
        QUERY["Query: [head_num, head_size]"]
        KV["KV Cache: [layer, seq_len, kv_dim]"]
        SCORE["Score: [head_num, seq_len]"]
    end

    subgraph 索引计算
        INDEX1["1D: index = i"]
        INDEX2["2D: index = i * dims_[1] + j"]
        INDEX3["3D: index = i*dims_[1]*dims_[2] + j*dims_[2] + k"]
        FLATTEN["size_ = 产品所有dims"]
    end

    D0 --> EMB
    D1 --> EMB
    EMB --> INDEX2
    KV --> INDEX3
```

## 4. Tensor数据类型

```mermaid
graph TB
    subgraph 数据类型
        FP32["kDataTypeFp32<br/>float, 4 bytes"]
        INT8["kDataTypeInt8<br/>int8_t, 1 byte"]
        INT32["kDataTypeInt32<br/>int32_t, 4 bytes"]
    end

    subgraph 使用场景
        FP32_USE["默认计算精度<br/>权重、激活值"]
        INT8_USE["量化权重<br/>Int8 MatMul"]
        INT32_USE["Token IDs<br/>位置索引"]
    end

    subgraph 类型转换
        CAST["类型转换"]
        QUANT["量化: FP32 → INT8"]
        DEQUANT["反量化: INT8 → FP32"]
    end

    FP32 --> FP32_USE
    INT8 --> INT8_USE
    INT32 --> INT32_USE

    FP32_USE --> QUANT
    QUANT --> INT8_USE
    INT8_USE --> DEQUANT
    DEQUANT --> FP32_USE
```

## 5. Tensor设备转移

```mermaid
sequenceDiagram
    participant CPU as CPU Memory
    participant Tensor as Tensor对象
    participant GPU as GPU Memory

    Note over Tensor: to_cuda()
    CPU->>Tensor: 获取CPU数据指针
    Tensor->>GPU: cudaMalloc分配GPU内存
    Tensor->>GPU: cudaMemcpyHostToDevice
    GPU-->>Tensor: 更新device_type_

    Note over Tensor: to_cpu()
    GPU->>Tensor: 获取GPU数据指针
    Tensor->>CPU: 确保CPU内存存在
    Tensor->>CPU: cudaMemcpyDeviceToHost
    CPU-->>Tensor: 更新device_type_
```

## 6. Buffer引用计数

```mermaid
graph TB
    subgraph Buffer创建
        NEW_BUF["创建新Buffer"]
        COUNT_1["引用计数 = 1"]
    end

    subgraph Tensor共享
        COPY_TENSOR["Tensor拷贝/赋值"]
        SHARED_PTR["shared_ptr拷贝"]
        COUNT_INC["引用计数 +1"]
    end

    subgraph Buffer释放
        TENSOR_DEL["Tensor析构"]
        COUNT_DEC["引用计数 -1"]
        CHECK_ZERO["检查是否为0"]
        FREE_MEM["释放内存"]
    end

    NEW_BUF --> COUNT_1
    COUNT_1 --> COPY_TENSOR
    COPY_TENSOR --> SHARED_PTR
    SHARED_PTR --> COUNT_INC

    COUNT_INC --> TENSOR_DEL
    TENSOR_DEL --> COUNT_DEC
    COUNT_DEC --> CHECK_ZERO
    CHECK_ZERO --> |引用计数=0| FREE_MEM
    CHECK_ZERO --> |引用计数>0| END["保留内存"]
```

## 7. ModelBufferType枚举

```mermaid
graph LR
    subgraph 输入Buffer
        INPUT["kInputTokens"]
        INPUT_EMB["kInputEmbedding"]
    end

    subgraph 位置Buffer
        POS["kPosTensor"]
    end

    subgraph 中间Buffer
        QUERY["kQuery"]
        KEY["kKeyCache"]
        VALUE["kValueCache"]
        SCORE["kScoreStorage"]
        MHA_OUT["kOutputMHA"]
        FFN_IN["kFFNInner"]
        FFN_OUT["kOutputFFN"]
        OUTPUT["kOutput"]
    end

    subgraph 输出Buffer
        LOGITS["kLogits"]
    end
```

## 8. Tensor在推理中的使用

```mermaid
sequenceDiagram
    participant Model as Model
    participant Buffer as BufferManager
    participant Tensor as Tensor
    participant Layer as Layer

    Model->>Buffer: get_buffer(type)
    Buffer-->>Tensor: 返回Tensor引用
    Model->>Layer: forward(Tensor)

    Note over Layer: Layer读取Tensor数据

    Layer->>Tensor: ptr<float>()
    Tensor-->>Layer: 返回数据指针

    Note over Layer: Layer写入结果到Tensor

    Layer-->>Model: Status::OK
    Model->>Buffer: 释放或复用Buffer
```

## 9. 外部内存Tensor

```mermaid
graph TB
    subgraph 场景
        MMAP["模型文件mmap"]
        SHARE["与其他模块共享"]
        ZERO_COPY["零拷贝需求"]
    end

    subgraph 创建方式
        EXT_PTR["传入外部指针"]
        EXT_BUF["use_external_=true"]
        NO_FREE["不自动释放"]
    end

    subgraph 注意事项
        LIFE_CYCLE["外部管理生命周期"]
        ALIGN["内存对齐要求"]
        SYNC["同步问题"]
    end

    MMAP --> EXT_PTR
    SHARE --> EXT_PTR
    ZERO_COPY --> EXT_PTR

    EXT_PTR --> EXT_BUF
    EXT_BUF --> NO_FREE

    NO_FREE --> LIFE_CYCLE
    NO_FREE --> ALIGN
    NO_FREE --> SYNC
```

## 10. Tensor vs raw pointer

```mermaid
graph LR
    subgraph Raw Pointer
        RAW["float* ptr"]
        RAW_ISSUE1["手动内存管理"]
        RAW_ISSUE2["无维度信息"]
        RAW_ISSUE3["无设备信息"]
    end

    subgraph Tensor
        TENSOR["Tensor类"]
        TENSOR_PRO1["自动内存管理"]
        TENSOR_PRO2["维度追踪"]
        TENSOR_PRO3["设备感知"]
        TENSOR_PRO4["类型安全"]
    end

    RAW --> |问题| RAW_ISSUE1
    RAW --> |问题| RAW_ISSUE2
    RAW --> |问题| RAW_ISSUE3

    TENSOR --> |优势| TENSOR_PRO1
    TENSOR --> |优势| TENSOR_PRO2
    TENSOR --> |优势| TENSOR_PRO3
    TENSOR --> |优势| TENSOR_PRO4
```
