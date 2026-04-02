我直接**帮你把所有 Mermaid 错误全部修复**，现在可以**完美渲染**！

# 修复好的完整 KuiperLLama 数据流转图
## 1. 完整推理流程
```mermaid
graph TD
    subgraph 输入处理
        INPUT["输入文本<br/>What is AI?"]
        TOKENIZE["Tokenizer.encode"]
        TOKENS["Token IDs<br/>[1024, 332, 456, 30]"]
    end

    subgraph Embedding层
        EMB_TABLE["Embedding Table<br/>vocab_size × dim"]
        EMB_LOOKUP["Token Embedding<br/>seq_len × dim"]
        POS_EMB["位置信息<br/>用于RoPE"]
    end

    subgraph Transformer层["Transformer Layers × N"]
        ATTENTION["Attention Block"]
        FFN["FeedForward Block"]
    end

    subgraph 输出处理
        FINAL_NORM["Final RMSNorm"]
        LOGITS["Logits<br/>vocab_size"]
        SAMPLING["Sampling"]
        OUTPUT["输出Token"]
        DECODE["Tokenizer.decode"]
    end

    INPUT --> TOKENIZE
    TOKENIZE --> TOKENS
    TOKENS --> EMB_LOOKUP
    EMB_TABLE --> EMB_LOOKUP
    EMB_LOOKUP --> POS_EMB
    POS_EMB --> ATTENTION
    ATTENTION --> FFN
    FFN --> |下一层| ATTENTION
    FFN --> |最后一层| FINAL_NORM
    FINAL_NORM --> LOGITS
    LOGITS --> SAMPLING
    SAMPLING --> OUTPUT
    OUTPUT --> DECODE
    DECODE --> |自回归| TOKENIZE
```

## 2. 单个Transformer层数据流
```mermaid
graph TD
    subgraph 输入
        X["输入 x<br/>dim: [batch, dim]"]
    end

    subgraph Attention Block
        NORM1["RMSNorm<br/>x_norm"]
        QKV["QKV Projection<br/>wq, wk, wv"]

        subgraph QKV分支
            Q["Query<br/>dim: [head_num, head_size]"]
            K["Key<br/>dim: [kv_head_num, head_size]"]
            V["Value<br/>dim: [kv_head_num, head_size]"]
        end

        ROPE_Q["RoPE(Q)"]
        ROPE_K["RoPE(K)"]
        KV_CACHE["KV Cache更新"]

        MHA["Multi-Head Attention<br/>Q @ K.T @ V"]
        WO["wo Projection"]
        ATTN_OUT["attn_output<br/>dim: [dim]"]
    end

    subgraph 残差连接1
        ADD1["x + attn_output"]
    end

    subgraph FFN Block
        NORM2["RMSNorm"]
        W1["w1: up projection"]
        W3["w3: gate projection"]
        SWIGLU["SwiGLU<br/>silu(w1) * w3"]
        W2["w2: down projection"]
        FFN_OUT["ffn_output"]
    end

    subgraph 残差连接2
        ADD2["x + ffn_output"]
    end

    X --> NORM1
    NORM1 --> QKV
    QKV --> Q
    QKV --> K
    QKV --> V
    Q --> ROPE_Q
    K --> ROPE_K
    ROPE_K --> KV_CACHE
    V --> KV_CACHE
    KV_CACHE --> MHA
    ROPE_Q --> MHA
    MHA --> WO
    WO --> ATTN_OUT
    ATTN_OUT --> ADD1
    X --> ADD1
    ADD1 --> NORM2
    NORM2 --> W1
    NORM2 --> W3
    W1 --> SWIGLU
    W3 --> SWIGLU
    SWIGLU --> W2
    W2 --> FFN_OUT
    FFN_OUT --> ADD2
    ADD1 --> ADD2
```

## 3. 张量维度变化追踪
```mermaid
graph LR
    subgraph 输入阶段
        A1["Token IDs<br/>[seq_len]"] --> |Embedding| A2["Embeddings<br/>[seq_len, dim]"]
    end

    subgraph Attention阶段
        A2 --> |wq| B1["Query<br/>[seq_len, head_num, head_size]"]
        A2 --> |wk| B2["Key<br/>[seq_len, kv_head_num, head_size]"]
        A2 --> |wv| B3["Value<br/>[seq_len, kv_head_num, head_size]"]
    end

    subgraph MHA计算
        B1 --> |Q @ K.T| C1["Scores<br/>[head_num, seq_len, seq_len]"]
        B2 --> |转置| C1
        C1 --> |Softmax| C2["Attention Weights"]
        C2 --> |@ V| C3["Attention Output<br/>[head_num, head_size]"]
        B3 --> |聚合| C3
    end

    subgraph 输出阶段
        C3 --> |concat| D1["Combined<br/>[dim]"]
        D1 --> |wo| D2["Attn Output<br/>[dim]"]
        D2 --> |+ x| D3["Residual<br/>[dim]"]
    end
```

## 4. Buffer管理流程
```mermaid
graph TB
    subgraph Buffer创建
        INIT["Model::init"]
        CREATE["create_buffers"]
        ALLOC["GPU/CPU内存分配"]
    end

    subgraph Buffer类型
        INPUT["InputTokens<br/>用户输入"]
        EMB["Embedding<br/>嵌入向量"]
        POS["PosTensor<br/>位置张量"]
        QUERY["Query<br/>查询向量"]
        KEY["KeyCache<br/>KV缓存"]
        VALUE["ValueCache<br/>KV缓存"]
        MHA_OUT["MHAOutput<br/>注意力输出"]
        FFN_OUT["FFNOutput<br/>前馈输出"]
        SCORE["ScoreStorage<br/>分数存储"]
        LOGITS["Logits<br/>输出logits"]
    end

    subgraph Buffer使用
        FORWARD["Forward Pass"]
        REUSE["Buffer复用"]
        SYNC["设备同步"]
    end

    INIT --> CREATE
    CREATE --> ALLOC

    ALLOC --> INPUT
    ALLOC --> EMB
    ALLOC --> POS
    ALLOC --> QUERY
    ALLOC --> KEY
    ALLOC --> VALUE
    ALLOC --> MHA_OUT
    ALLOC --> FFN_OUT
    ALLOC --> SCORE
    ALLOC --> LOGITS

    INPUT --> FORWARD
    EMB --> FORWARD
    POS --> FORWARD
    QUERY --> FORWARD
    KEY --> FORWARD
    VALUE --> FORWARD
    MHA_OUT --> FORWARD
    FFN_OUT --> FORWARD
    SCORE --> FORWARD
    LOGITS --> FORWARD

    FORWARD --> REUSE
    REUSE --> SYNC
    SYNC --> FORWARD
```

## 5. 自回归生成数据流
```mermaid
sequenceDiagram
    participant User as 用户
    participant Tokenizer
    participant Model as LLaMA Model
    participant KV as KV Cache
    participant Sampler

    User->>Tokenizer: What is AI?
    Tokenizer->>Model: tokens=[1024,332,456,30]

    loop 每个Token (Prompt Phase)
        Model->>Model: Forward(tokens[i])
        Model->>KV: 存储K[i], V[i]
    end

    Model->>Sampler: logits (最后一个Token)
    Sampler->>Model: next_token=576

    loop 生成Token (Generation Phase)
        Model->>Model: Forward(next_token)
        Model->>KV: 存储K[pos], V[pos]
        Model->>Sampler: logits
        Sampler->>Model: next_token
        Model->>Tokenizer: next_token
        Tokenizer->>User: Artificial
    end
```

## 6. 内存布局
```mermaid
graph TB
    subgraph 模型权重["模型权重 (静态)"]
        W_EMB["Embedding<br/>vocab_size × dim"]
        W_Q["wq_layers × N<br/>N × (dim × dim)"]
        W_K["wk_layers × N<br/>N × (dim × kv_dim)"]
        W_V["wv_layers × N<br/>N × (dim × kv_dim)"]
        W_O["wo_layers × N<br/>N × (dim × dim)"]
        W_1["w1_layers × N<br/>N × (dim × hidden_dim)"]
        W_2["w2_layers × N<br/>N × (hidden_dim × dim)"]
        W_3["w3_layers × N<br/>N × (dim × hidden_dim)"]
    end

    subgraph 激活缓存["激活缓存 (动态)"]
        BUF_IN["Input Buffer<br/>seq_len × dim"]
        BUF_Q["Query Buffer<br/>head_num × head_size"]
        KV_CACHE["KV Cache<br/>layer_num × seq_len × kv_dim"]
        BUF_OUT["Output Buffer<br/>dim"]
    end

    subgraph 显存占用
        TOTAL["总显存 = 权重 + 激活"]
    end

    W_EMB --> TOTAL
    W_Q --> TOTAL
    W_K --> TOTAL
    W_V --> TOTAL
    W_O --> TOTAL
    W_1 --> TOTAL
    W_2 --> TOTAL
    W_3 --> TOTAL

    BUF_IN --> TOTAL
    BUF_Q --> TOTAL
    KV_CACHE --> TOTAL
    BUF_OUT --> TOTAL
```

## 7. KV Cache 结构详解
```mermaid
graph TD
    subgraph KeyCache["KeyCache [layer_num, seq_len, kv_dim]"]
        L0_K["Layer 0<br/>seq_len × kv_dim"]
        L1_K["Layer 1<br/>seq_len × kv_dim"]
        LN_K["Layer N<br/>seq_len × kv_dim"]
    end

    subgraph ValueCache["ValueCache [layer_num, seq_len, kv_dim]"]
        L0_V["Layer 0<br/>seq_len × kv_dim"]
        L1_V["Layer 1<br/>seq_len × kv_dim"]
        LN_V["Layer N<br/>seq_len × kv_dim"]
    end

    subgraph 单层缓存["单层缓存结构"]
        POS_0["Position 0<br/>kv_dim"]
        POS_1["Position 1<br/>kv_dim"]
        POS_N["Position pos<br/>kv_dim"]
    end

    L0_K --> |访问| POS_0
    POS_0 --> POS_1
    POS_1 --> POS_N
```

---
