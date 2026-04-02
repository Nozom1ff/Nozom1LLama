# KuiperLLama CUDA执行流程图

## 1. CUDA Kernel总览

```mermaid
graph TB
    subgraph 计算类Kernel
        MATMUL["MatMul Kernel<br/>矩阵乘法"]
        MHA["MHA Kernel<br/>多头注意力"]
        RMS["RMSNorm Kernel<br/>归一化"]
        ROPE["RoPE Kernel<br/>位置编码"]
        SOFTMAX["Softmax Kernel<br/>概率归一化"]
        SWIGLU["SwiGLU Kernel<br/>激活函数"]
        ADD["Add Kernel<br/>元素加法"]
    end

    subgraph 内存类Kernel
        MEMCPY["Memcpy Kernel<br/>数据传输"]
        FILL["Fill Kernel<br/>填充数据"]
        EMBED["Embedding Kernel<br/>查表"]
    end

    subgraph 量化类Kernel
        QUANT["Quantize Kernel<br/>量化"]
        DEQUANT["Dequantize Kernel<br/>反量化"]
    end
```

## 2. CUDA Kernel调度流程

```mermaid
sequenceDiagram
    participant Host as CPU Host
    participant API as CUDA API
    participant Stream as CUDA Stream
    participant GPU as GPU Device

    Host->>API: cudaSetDevice(0)
    Host->>API: cudaStreamCreate(&stream)

    Note over Host: 准备Kernel参数

    Host->>API: kernel<<<grid, block, shared_mem, stream>>>(args)
    API->>Stream: 将Kernel加入队列
    Stream->>GPU: 异步执行Kernel

    Note over GPU: Kernel执行中...

    Host->>API: cudaStreamSynchronize(stream)
    API->>Stream: 等待完成
    Stream-->>Host: 执行完成
```

## 3. MHA CUDA Kernel详解

```mermaid
graph TD
    subgraph Kernel配置
        GRID["Grid: head_num blocks<br/>例: 32 blocks"]
        BLOCK["Block: 256 threads"]
        SHARED["Shared Memory: head_size × 4 bytes<br/>例: 128 × 4 = 512 bytes"]
    end

    subgraph 执行模型
        HEAD["每个Block处理一个Head"]
        THREAD["每个Thread处理多个元素"]
        COOP["Thread间协作同步"]
    end

    subgraph 内存层次
        GLOBAL["Global Memory<br/>KeyCache, ValueCache, Output"]
        SHARED_MEM["Shared Memory<br/>Query缓存"]
        REG["Registers<br/>局部计算"]
    end

    GRID --> HEAD
    BLOCK --> THREAD
    SHARED --> COOP

    HEAD --> GLOBAL
    COOP --> SHARED_MEM
    THREAD --> REG
```

## 4. MHA Kernel执行步骤

```mermaid
graph TD
    subgraph Step1["Step 1: 加载Query到Shared Memory"]
        S1_START["每个线程加载一部分Query"]
        S1_LOOP["for i = threadIdx.x; i < head_size; i += blockDim.x"]
        S1_LOAD["s_query[i] = query[i]"]
        S1_SYNC["__syncthreads()"]
    end

    subgraph Step2["Step 2: 计算Attention Score"]
        S2_LOOP["for t = threadIdx.x; t <= pos; t += blockDim.x"]
        S2_LOAD_K["加载KeyCache[t] (float4向量化)"]
        S2_LOAD_Q["加载s_query (float4向量化)"]
        S2_DOT["点积计算: score += key × query"]
        S2_SCALE["score *= scale (1/√head_size)"]
        S2_STORE["score_head[t] = score"]
    end

    subgraph Step3["Step 3: Softmax"]
        S3_FIND_MAX["warp级归约找最大值"]
        S3_EXP["计算 exp(score - max)"]
        S3_SUM["warp级归约求和"]
        S3_NORM["score = exp / sum"]
    end

    subgraph Step4["Step 4: 加权求和Value"]
        S4_LOOP["for i = threadIdx.x; i < head_size; i += blockDim.x"]
        S4_ACC["value = 0"]
        S4_INNER["for t = 0; t <= pos; t++"]
        S4_LOAD_V["加载ValueCache[t][i]"]
        S4_MUL["value += score[t] × value_cache[t][i]"]
        S4_STORE["output[i] = value"]
    end

    S1_START --> S1_LOOP
    S1_LOOP --> S1_LOAD
    S1_LOAD --> S1_SYNC
    S1_SYNC --> S2_LOOP
    S2_LOOP --> S2_LOAD_K
    S2_LOAD_K --> S2_LOAD_Q
    S2_LOAD_Q --> S2_DOT
    S2_DOT --> S2_SCALE
    S2_SCALE --> S2_STORE
    S2_STORE --> S3_FIND_MAX
    S3_FIND_MAX --> S3_EXP
    S3_EXP --> S3_SUM
    S3_SUM --> S3_NORM
    S3_NORM --> S4_LOOP
    S4_LOOP --> S4_ACC
    S4_ACC --> S4_INNER
    S4_INNER --> S4_LOAD_V
    S4_LOAD_V --> S4_MUL
    S4_MUL --> S4_STORE
```

## 5. float4向量化优化

```mermaid
graph LR
    subgraph 传统加载
        OLD["逐个float加载"]
        OLD_LOOP["for i = 0; i < 128; i++"]
        OLD_LOAD["float val = ptr[i]"]
        OLD_COUNT["128次内存事务"]
    end

    subgraph float4向量化
        NEW["float4批量加载"]
        NEW_LOOP["for i = 0; i < 128; i += 4"]
        NEW_LOAD["float4 val = *(float4*)(ptr + i)"]
        NEW_COUNT["32次内存事务"]
    end

    subgraph 性能提升
        BANDWIDTH["内存带宽提升 4×"]
        LATENCY["延迟减少"]
    end

    OLD --> |优化| NEW
    NEW --> BANDWIDTH
    NEW --> LATENCY
```

## 6. Warp级并行

```mermaid
graph TB
    subgraph Warp结构
        WARP["一个Warp = 32个线程"]
        SIMT["SIMT执行模型"]
        LOCKSTEP["锁步执行"]
    end

    subgraph Softmax Warp归约
        MAX_INIT["每个线程持有局部最大值"]
        SHFL_MAX["__shfl_down_sync 归约"]
        MAX_RESULT["Warp内最大值"]

        SUM_INIT["每个线程持有局部和"]
        SHFL_SUM["__shfl_down_sync 归约"]
        SUM_RESULT["Warp内总和"]
    end

    subgraph 优势
        NO_SMEM["不使用Shared Memory"]
        FAST["Warp内通信快速"]
    end

    WARP --> MAX_INIT
    MAX_INIT --> SHFL_MAX
    SHFL_MAX --> MAX_RESULT
    MAX_RESULT --> SUM_INIT
    SUM_INIT --> SHFL_SUM
    SHFL_SUM --> SUM_RESULT

    SHFL_MAX --> NO_SMEM
    SHFL_SUM --> FAST
```

## 7. CUDA内存层次优化

```mermaid
graph TB
    subgraph 内存层次
        REGS["Registers<br/>最快, 每线程私有"]
        SHARED["Shared Memory<br/>快速, Block内共享"]
        L1["L1 Cache<br/>SM内缓存"]
        L2["L2 Cache<br/>GPU全局缓存"]
        GLOBAL["Global Memory<br/>最慢, 所有SM共享"]
    end

    subgraph 优化策略
        REG_USE["局部变量放寄存器"]
        SMEM_USE["复用数据放Shared Memory"]
        COALESCE["合并访问Global Memory"]
        VECTOR["向量化加载(float4)"]
    end

    REGS --> |最快| REG_USE
    SHARED --> |快速| SMEM_USE
    GLOBAL --> |合并| COALESCE
    COALESCE --> VECTOR

    REGS -.-> SHARED
    SHARED -.-> L1
    L1 -.-> L2
    L2 -.-> GLOBAL
```

## 8. RMSNorm CUDA Kernel

```mermaid
graph TD
    subgraph 输入
        INPUT["input tensor [dim]"]
        WEIGHT["weight [dim]"]
    end

    subgraph Step1: 计算平方和
        SQ_LOOP["for i = 0; i < dim; i++"]
        SQ_ACC["ss += input[i] × input[i]"]
        SQ_REDUCE["Block级归约求和"]
    end

    subgraph Step2: 归一化
        MEAN["mean = ss / dim"]
        RMS["rms = 1 / √(mean + eps)"]
    end

    subgraph Step3: 缩放
        OUT_LOOP["for i = 0; i < dim; i++"]
        OUT_CALC["output[i] = weight[i] × (input[i] × rms)"]
    end

    subgraph 输出
        OUTPUT["output tensor [dim]"]
    end

    INPUT --> SQ_LOOP
    SQ_LOOP --> SQ_ACC
    SQ_ACC --> SQ_REDUCE
    SQ_REDUCE --> MEAN
    MEAN --> RMS
    RMS --> OUT_LOOP
    WEIGHT --> OUT_LOOP
    OUT_LOOP --> OUT_CALC
    OUT_CALC --> OUTPUT
```

## 9. RoPE CUDA Kernel

```mermaid
graph TD
    subgraph 输入
        QK["Query或Key [head_num, head_size]"]
        POS["位置pos"]
        FREQ["频率表freq_cache"]
    end

    subgraph RoPE计算
        PAIR["head_size内相邻两个为一组"]
        ANGLE["angle = pos × freq[i]"]
        COS_VAL["cos_val = cos(angle)"]
        SIN_VAL["sin_val = sin(angle)"]

        ROTATE["旋转计算:<br/>out[2i] = in[2i]×cos - in[2i+1]×sin<br/>out[2i+1] = in[2i]×sin + in[2i+1]×cos"]
    end

    subgraph 输出
        ROTATED["旋转后的Q或K"]
    end

    QK --> PAIR
    POS --> ANGLE
    FREQ --> ANGLE
    ANGLE --> COS_VAL
    ANGLE --> SIN_VAL
    COS_VAL --> ROTATE
    SIN_VAL --> ROTATE
    PAIR --> ROTATE
    ROTATE --> ROTATED
```

## 10. Kernel性能对比

```mermaid
graph LR
    subgraph CPU实现
        CPU_MATMUL["MatMul: ~100ms"]
        CPU_MHA["MHA: ~500ms"]
        CPU_RMS["RMSNorm: ~10ms"]
    end

    subgraph CUDA实现
        GPU_MATMUL["MatMul: ~5ms<br/>20×加速"]
        GPU_MHA["MHA: ~20ms<br/>25×加速"]
        GPU_RMS["RMSNorm: ~0.5ms<br/>20×加速"]
    end

    CPU_MATMUL --> |GPU加速| GPU_MATMUL
    CPU_MHA --> |GPU加速| GPU_MHA
    CPU_RMS --> |GPU加速| GPU_RMS
```
