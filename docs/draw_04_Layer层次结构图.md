# KuiperLLama Layer层次结构图

## 1. Layer类型总览

```mermaid
graph TB
    subgraph 基类
        BASE[BaseLayer<br/>抽象基类]
        LAYER[Layer<br/>实现输入输出管理]
        PARAM[LayerParam<br/>带参数的Layer]
    end

    subgraph 注意力相关Layer
        EMB[Embedding<br/>kLayerEmbedding]
        WQ[Linear/wq<br/>kLayerLinear]
        WK[Linear/wk<br/>kLayerLinear]
        WV[Linear/wv<br/>kLayerLinear]
        WO[Linear/wo<br/>kLayerLinear]
        ROPE[RoPE<br/>kLayerRoPe]
        MHA[MultiHeadAttention<br/>kLayerMHA]
    end

    subgraph 归一化Layer
        RMS[RMSNorm<br/>kLayerRMSNorm]
        RMS_Q[RMSNorm<br/>query归一化]
        RMS_F[RMSNorm<br/>FFN归一化]
    end

    subgraph 前馈网络Layer
        W1[Linear/w1<br/>kLayerLinear]
        W2[Linear/w2<br/>kLayerLinear]
        W3[Linear/w3<br/>kLayerLinear]
        SWIGLU[SwiGLU<br/>kLayerSwiGLU]
    end

    subgraph 输出Layer
        ADD[Add<br/>kLayerAdd]
        FINAL[Linear/final<br/>kLayerLinear]
    end

    BASE --> LAYER
    LAYER --> PARAM
    LAYER --> ROPE
    LAYER --> MHA
    LAYER --> ADD
    LAYER --> SWIGLU

    PARAM --> EMB
    PARAM --> WQ
    PARAM --> WK
    PARAM --> WV
    PARAM --> WO
    PARAM --> RMS
    PARAM --> W1
    PARAM --> W2
    PARAM --> W3
    PARAM --> FINAL
```

## 2. LLaMA模型Layer组织结构

```mermaid
graph TD
    subgraph LLama2Model
        EMBED[embedding_layer_<br/>词嵌入层]

        subgraph 每层独立权重
            L0_WQ[wq_layers_[0]]
            L0_WK[wk_layers_[0]]
            L0_WV[wv_layers_[0]]
            L0_WO[wo_layers_[0]]
            L0_RMS_Q[rms_attn_layers_[0]]
            L0_RMS_F[rms_ffn_layers_[0]]
            L0_W1[w1_layers_[0]]
            L0_W2[w2_layers_[0]]
            L0_W3[w3_layers_[0]]

            L1_WQ[wq_layers_[1]]
            L1_WK[wk_layers_[1]]
            ...
        end

        subgraph 共享Layer
            MHA_SHARED[mha_layer_<br/>所有层共享]
            ROPE_SHARED[rope_layer_<br/>所有层共享]
        end

        FINAL_RMS[rms_final_layer_]
        FINAL_LINEAR[norm_linear_layer_]
    end

    EMBED --> L0_WQ
    L0_WQ --> MHA_SHARED
    MHA_SHARED --> L0_WO
    L0_WO --> L0_RMS_F
    L0_RMS_F --> L0_W1
    L0_W1 --> L0_W2
    L0_W3 --> L0_W2
    L0_W2 --> L1_WQ

    MHA_SHARED -.-> |layer_idx参数| L0_WQ
    MHA_SHARED -.-> |layer_idx参数| L1_WQ
```

## 3. Transformer Block内部结构

```mermaid
graph TD
    subgraph Input
        X[输入 x<br/>[batch, dim]]
    end

    subgraph "Attention Block (Pre-Norm)"
        NORM1[RMSNorm]
        QKV_PROJ[QKV Projection]

        subgraph QKV分支
            WQ[wq: x → Q<br/>[dim] → [head_num × head_size]]
            WK[wk: x → K<br/>[dim] → [kv_head_num × head_size]]
            WV[wv: x → V<br/>[dim] → [kv_head_num × head_size]]
        end

        ROPE[RoPE位置编码]
        KV_UPDATE[KV Cache更新]
        MHA_CALC[MHA计算]
        WO_PROJ[wo投影]
        RES1[残差连接: x + attn]
    end

    subgraph "FFN Block (Pre-Norm)"
        NORM2[RMSNorm]

        subgraph SwiGLU分支
            UP[w1: up projection<br/>[dim] → [hidden_dim]]
            GATE[w3: gate projection<br/>[dim] → [hidden_dim]]
            ACT[silu(up) × gate]
            DOWN[w2: down projection<br/>[hidden_dim] → [dim]]
        end

        RES2[残差连接: x + ffn]
    end

    subgraph Output
        OUT[输出<br/>[batch, dim]]
    end

    X --> NORM1
    NORM1 --> WQ
    NORM1 --> WK
    NORM1 --> WV
    WQ --> ROPE
    WK --> ROPE
    ROPE --> KV_UPDATE
    WV --> KV_UPDATE
    KV_UPDATE --> MHA_CALC
    MHA_CALC --> WO_PROJ
    WO_PROJ --> RES1
    X --> RES1

    RES1 --> NORM2
    NORM2 --> UP
    NORM2 --> GATE
    UP --> ACT
    GATE --> ACT
    ACT --> DOWN
    DOWN --> RES2
    RES1 --> RES2

    RES2 --> OUT
```

## 4. Layer参数矩阵维度

```mermaid
graph LR
    subgraph 模型配置
        DIM[dim = 4096]
        HIDDEN[hidden_dim = 11008]
        HEAD[head_num = 32]
        HEAD_SIZE[head_size = 128]
        KV_HEAD[kv_head_num = 4]
        KV_DIM[kv_dim = 512]
    end

    subgraph Attention权重
        WQ_MAT["wq: [dim, dim]<br/>4096 × 4096"]
        WK_MAT["wk: [dim, kv_dim]<br/>4096 × 512"]
        WV_MAT["wv: [dim, kv_dim]<br/>4096 × 512"]
        WO_MAT["wo: [dim, dim]<br/>4096 × 4096"]
    end

    subgraph FFN权重
        W1_MAT["w1: [dim, hidden_dim]<br/>4096 × 11008"]
        W2_MAT["w2: [hidden_dim, dim]<br/>11008 × 4096"]
        W3_MAT["w3: [dim, hidden_dim]<br/>4096 × 11008"]
    end

    subgraph 归一化权重
        RMS_W["rms_weight: [dim]<br/>4096"]
    end

    DIM --> WQ_MAT
    DIM --> WK_MAT
    KV_DIM --> WK_MAT
    DIM --> WV_MAT
    KV_DIM --> WV_MAT
    DIM --> WO_MAT
    DIM --> W1_MAT
    HIDDEN --> W1_MAT
    HIDDEN --> W2_MAT
    DIM --> W2_MAT
    DIM --> W3_MAT
    HIDDEN --> W3_MAT
    DIM --> RMS_W
```

## 5. LayerForward调用顺序

```mermaid
sequenceDiagram
    participant Model as LLama2Model
    participant RMS1 as RMSNorm
    participant WQ as wq_layer
    participant WK as wk_layer
    participant WV as wv_layer
    participant RoPE as RoPE_layer
    participant MHA as MHA_layer
    participant WO as wo_layer
    participant RMS2 as RMSNorm
    participant W1 as w1_layer
    participant W3 as w3_layer
    participant W2 as w2_layer

    Model->>RMS1: forward(x) → x_norm
    Note over RMS1: attention_rms()

    Model->>WQ: forward(x_norm) → Q
    Model->>WK: forward(x_norm) → K
    Model->>WV: forward(x_norm) → V
    Note over WQ,WV: attention_qkv()

    Model->>RoPE: forward(Q, pos) → Q_rot
    Model->>RoPE: forward(K, pos) → K_rot
    Note over RoPE: 位置编码

    Model->>MHA: forward(Q, K_cache, V_cache)
    Note over MHA: 更新KV Cache<br/>计算注意力

    Model->>WO: forward(mha_out) → attn_out
    Note over WO: attention_mha()

    Model->>RMS2: forward(x + attn_out)
    Model->>W1: forward(ffn_in) → up
    Model->>W3: forward(ffn_in) → gate
    Note over W1,W3: feed_forward()

    Model->>W2: forward(silu(up) × gate)
    Note over W2: FFN输出
```

## 6. 共享Layer vs 独立Layer

```mermaid
graph TB
    subgraph 每层独立的权重
        INDEP[每层有独立的权重矩阵]
        WQ_L0[wq_layers_[0]]
        WQ_L1[wq_layers_[1]]
        WQ_LN[wq_layers_[N-1]]
        WK_ALL[wk/wv/wo同理]
        W1_ALL[w1/w2/w3同理]
        RMS_ALL[rms同理]
    end

    subgraph 所有层共享的计算
        SHARED[所有层共享计算逻辑]
        MHA_ALL[mha_layer_<br/>通过set_layer_idx区分层]
        ROPE_ALL[rope_layer_<br/>通过pos参数区分]
    end

    subgraph 设计原因
        REASON1[权重不同: 每层学习不同特征]
        REASON2[计算相同: MHA/RoPE算法一致]
        REASON3[节省内存: 不重复存储相同逻辑]
    end

    INDEP --> WQ_L0
    INDEP --> WQ_L1
    INDEP --> WQ_LN

    SHARED --> MHA_ALL
    SHARED --> ROPE_ALL

    REASON1 --> INDEP
    REASON2 --> SHARED
    REASON3 --> SHARED
```

## 7. Layer初始化流程

```mermaid
graph TD
    subgraph 模型初始化
        INIT[init_model]
        READ[读取模型配置]
        ALLOC[分配Layer存储空间]
    end

    subgraph 创建参数层
        EMB_LAY[创建Embedding层]
        ATTN_LAY[创建Attention权重层]
        FFN_LAY[创建FFN权重层]
        FINAL_LAY[创建Final层]
    end

    subgraph 创建非参数层
        MHA_LAY[创建MHA层<br/>共享]
        ROPE_LAY[创建RoPE层<br/>共享]
    end

    subgraph 加载权重
        LOAD_EMB[加载embedding权重]
        LOAD_ATTN[加载wq/wk/wv/wo权重]
        LOAD_FFN[加载w1/w2/w3权重]
        LOAD_FINAL[加载final权重]
    end

    INIT --> READ
    READ --> ALLOC
    ALLOC --> EMB_LAY
    ALLOC --> ATTN_LAY
    ALLOC --> FFN_LAY
    ALLOC --> FINAL_LAY
    ALLOC --> MHA_LAY
    ALLOC --> ROPE_LAY

    EMB_LAY --> LOAD_EMB
    ATTN_LAY --> LOAD_ATTN
    FFN_LAY --> LOAD_FFN
    FINAL_LAY --> LOAD_FINAL
```
