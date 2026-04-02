# KuiperLLama KV Cache管理图

## 1. KV Cache整体结构

```mermaid
graph TB
    subgraph KeyCache ["KeyCache [layer_num, seq_len, kv_dim]"]
        KC_L0["Layer 0: [seq_len, kv_dim]"]
        KC_L1["Layer 1: [seq_len, kv_dim]"]
        KC_DOTS["..."]
        KC_LN["Layer N-1: [seq_len, kv_dim]"]
    end

    subgraph ValueCache ["ValueCache [layer_num, seq_len, kv_dim]"]
        VC_L0["Layer 0: [seq_len, kv_dim]"]
        VC_L1["Layer 1: [seq_len, kv_dim]"]
        VC_DOTS["..."]
        VC_LN["Layer N-1: [seq_len, kv_dim]"]
    end

    subgraph 单层缓存布局
        POS_0["pos=0: [kv_dim]"]
        POS_1["pos=1: [kv_dim]"]
        POS_P["pos=p: [kv_dim]"]
        POS_S["pos=seq_len-1: [kv_dim]"]
    end

    KC_L0 --> POS_0
```

## 2. KV Cache更新流程

```mermaid
sequenceDiagram
    participant Model as LLama2Model
    participant QKV as attention_qkv()
    participant RoPE as RoPE Layer
    participant Cache as KV Cache
    participant MHA as MHA Layer

    Note over Model: 当前位置 pos

    Model->>QKV: 计算 Q, K, V
    QKV->>RoPE: 对 K 应用 RoPE
    RoPE-->>Model: K_rotated

    Model->>Cache: key_cache[layer_idx, pos] = K_rotated
    Model->>Cache: value_cache[layer_idx, pos] = V

    Note over Cache: KV Cache已更新

    Model->>MHA: 传入 query, key_cache, value_cache
    MHA->>Cache: 读取 key_cache[0:pos+1]
    MHA->>Cache: 读取 value_cache[0:pos+1]

    Note over MHA: 计算注意力<br/>output = softmax(Q·K.T)·V
```

## 3. KV Cache内存布局

```mermaid
graph LR
    subgraph 连续内存布局
        MEM["一维连续内存<br/>layer_num × seq_len × kv_dim floats"]

        subgraph Layer0区域
            L0_START["偏移: 0"]
            L0_DATA["Layer 0 数据"]
            L0_END["偏移: seq_len × kv_dim"]
        end

        subgraph Layer1区域
            L1_START["偏移: seq_len × kv_dim"]
            L1_DATA["Layer 1 数据"]
        end

        subgraph LayerN区域
            LN_START["偏移: (N-1) × seq_len × kv_dim"]
            LN_DATA["Layer N-1 数据"]
        end
    end

    subgraph 访问计算
        FORMULA["offset = layer_idx × seq_len × kv_dim<br/>+ pos × kv_dim"]
    end

    MEM --> L0_START
    MEM --> L1_START
    MEM --> LN_START
    L0_END --> FORMULA
```

## 4. Prefill vs Decode阶段

```mermaid
graph TB
    subgraph Prefill阶段["Prefill阶段 (处理Prompt)"]
        P1[Token 0]
        P2[Token 1]
        P3[Token 2]
        PN[Token N-1]

        P1 --> |位置0| PC1[KV Cache填充]
        P2 --> |位置1| PC2[KV Cache填充]
        P3 --> |位置2| PC3[KV Cache填充]
        PN --> |位置N-1| PCN[KV Cache填充]

        Note1["并行处理所有Prompt Token<br/>或串行处理每个Token"]
    end

    subgraph Decode阶段["Decode阶段 (生成Token)"]
        D1[Token N]
        D2[Token N+1]
        D3[Token N+2]

        D1 --> |位置N| DC1[KV Cache追加]
        D2 --> |位置N+1| DC2[KV Cache追加]
        D3 --> |位置N+2| DC3[KV Cache追加]

        Note2["每生成一个Token<br/>追加到KV Cache末尾"]
    end

    PC1 --> D1
    PCN --> D1
```

## 5. KV Cache访问模式

```mermaid
graph TD
    subgraph 写入模式
        WRITE["写入: 当前位置的 K 和 V"]
        W_OFFSET["offset = layer_idx × seq_len × kv_dim + pos × kv_dim"]
        W_DATA["写入 K[pos] 和 V[pos]"]
    end

    subgraph 读取模式
        READ["读取: 所有历史位置的 K 和 V"]
        R_LOOP["遍历 t = 0 to pos"]
        R_OFFSET["offset = layer_idx × seq_len × kv_dim + t × kv_dim"]
        R_DATA["读取 K[t] 和 V[t]"]
    end

    subgraph MHA使用
        MHA_READ["MHA需要读取所有历史KV"]
        SCORE["计算: score[t] = Q · K[t]"]
        AGGREGATE["聚合: output += softmax(score) × V[t]"]
    end

    WRITE --> W_OFFSET
    W_OFFSET --> W_DATA

    READ --> R_LOOP
    R_LOOP --> R_OFFSET
    R_OFFSET --> R_DATA

    R_DATA --> MHA_READ
    MHA_READ --> SCORE
    SCORE --> AGGREGATE
```

## 6. KV Cache显存计算

```mermaid
graph LR
    subgraph 计算公式
        FORMULA["KV Cache 大小 =<br/>2 × layer_num × seq_len × kv_dim × sizeof(float)"]
    end

    subgraph LLaMA-7B示例
        L7B_PARAM["layer_num = 32<br/>seq_len = 2048<br/>kv_dim = 512 (GQA)<br/>float = 4 bytes"]
        L7B_CALC["2 × 32 × 2048 × 512 × 4"]
        L7B_RESULT["= 256 MB"]
    end

    subgraph LLaMA-70B示例
        L70B_PARAM["layer_num = 80<br/>seq_len = 4096<br/>kv_dim = 1024<br/>float = 4 bytes"]
        L70B_CALC["2 × 80 × 4096 × 1024 × 4"]
        L70B_RESULT["= 2.5 GB"]
    end

    subgraph 优化策略
        GQA["GQA: kv_dim = head_size × kv_head_num<br/>减少 KV 头数量"]
        QUANT["量化: float16/int8<br/>减少每个元素大小"]
        PAGED["PagedAttention<br/>按需分配"]
    end

    FORMULA --> L7B_PARAM
    L7B_PARAM --> L7B_CALC
    L7B_CALC --> L7B_RESULT

    FORMULA --> L70B_PARAM
    L70B_PARAM --> L70B_CALC
    L70B_CALC --> L70B_RESULT

    L7B_RESULT --> GQA
    GQA --> QUANT
    QUANT --> PAGED
```

## 7. GQA对KV Cache的影响

```mermaid
graph TB
    subgraph 标准MHA
        MHA_HEAD["head_num = 32<br/>所有头都有独立KV"]
        MHA_DIM["kv_dim = head_size × head_num<br/>= 128 × 32 = 4096"]
        MHA_CACHE["KV Cache 较大"]
    end

    subgraph GQA-8
        GQA_HEAD["head_num = 32<br/>kv_head_num = 4"]
        GQA_DIM["kv_dim = head_size × kv_head_num<br/>= 128 × 4 = 512"]
        GQA_CACHE["KV Cache 减少 8×"]
    end

    subgraph 映射关系
        MAP["Query h0-h3 → KV H0<br/>Query h4-h7 → KV H1<br/>Query h8-h11 → KV H2<br/>Query h12-h15 → KV H3"]
    end

    MHA_HEAD --> MHA_DIM
    MHA_DIM --> MHA_CACHE

    GQA_HEAD --> GQA_DIM
    GQA_DIM --> GQA_CACHE

    MHA_CACHE --> |GQA优化| GQA_CACHE
    GQA_HEAD --> MAP
```

## 8. KV Cache代码实现

```mermaid
graph TD
    subgraph Buffer类型定义
        BUF_KEY["ModelBufferType::kKeyCache"]
        BUF_VAL["ModelBufferType::kValueCache"]
    end

    subgraph Buffer创建
        CREATE["create_buffers()"]
        SIZE["size = layer_num × seq_len × kv_dim"]
        ALLOC["cudaMalloc or malloc"]
    end

    subgraph Cache访问
        GET["get_buffer(kKeyCache)"]
        INDEX["通过layer_idx和pos索引"]
        OFFSET["layer_offset = layer_idx × seq_len × kv_dim<br/>pos_offset = pos × kv_dim"]
    end

    subgraph MHA使用
        MHA_GET["MultiHeadAttention::forward()"]
        KERNEL["mha_kernel访问缓存"]
    end

    BUF_KEY --> CREATE
    BUF_VAL --> CREATE
    CREATE --> SIZE
    SIZE --> ALLOC

    ALLOC --> GET
    GET --> INDEX
    INDEX --> OFFSET

    OFFSET --> MHA_GET
    MHA_GET --> KERNEL
```

## 9. KV Cache生命周期

```mermaid
graph LR
    subgraph 初始化
        INIT["模型初始化时分配"]
        MEM_ALLOC["分配最大seq_len空间"]
    end

    subgraph 推理过程
        PREFILL["Prefill: 填充Prompt的KV"]
        DECODE["Decode: 逐个追加新KV"]
    end

    subgraph 推理结束
        FREE["释放KV Cache内存"]
    end

    INIT --> MEM_ALLOC
    MEM_ALLOC --> PREFILL
    PREFILL --> DECODE
    DECODE --> |生成完成| FREE
```
