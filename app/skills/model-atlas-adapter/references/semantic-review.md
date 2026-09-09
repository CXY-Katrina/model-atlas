# 公式、源码、参数与图示检视

主 agent 用本文件建立研究证据；语义 agent 从原始输入独立核验，不能只相信主 agent 的摘要。覆盖所选范围中全部节点／边、公式、符号、权重和可见数值。

核验链包括 `docs/models/<model-id>/` 的模型知识成果、Archify typed JSON、对应生成 HTML/SVG 和实际页面。任务、review 与答复属于 `.scratch/` 协作记录，不能当作上游事实证据或混入读者的模型介绍。跨图分层时核对稳定 id 映射和完整依赖，不能为简化 Archify 布局而遗漏必要计算。

## S1. 版本与后端

- 记录仓库、固定 SHA、文件、类／函数和具体代码位置；行号只是辅助，固定提交和 symbol 才是长期定位依据。粘贴代码记录哈希与缺失依赖。
- 用户只提供一种实现时不强制引入另一后端；如果展示 vLLM 与 Transformers 对照，分别阅读各自完整依赖，标清两者版本和实现差异。
- 模型配置、checkpoint 变体、层范围、模态与权重必须属于同一适配目标。数据来源不同版本时显式说明兼容依据；不把 main 的新代码配成旧 checkpoint 的已证实行为。
- 论文／参考图帮助理解，但源码、配置和所选 checkpoint 决定这里展示的实现。无法确认的说法写“未验证”并从确定性路径中移除，关键缺口阻塞验收。

## S2. 算子范围与真实执行

- 每个图示 OP 说明实际执行范围，并绑定对应代码。QKV／Index projection 的详情应解释投影，不能把下游 attention forward 当作该投影的实现证据。
- 沿调用链追踪 packed projection 的所有分量及切分维度。若图宣称一次 GEMM 产生 Q/K/V/Qidx/Kidx，证据必须覆盖全部输出和权重排列；只看到主 QKV 投影不够。
- 区分 GEMM 投影、norm、RoPE、cache 写入和 attention backend。函数名包含 fused 不等于融合了所有这些操作；核对参数、调用者和内核实际代码。
- 概念图中的 QK、scale、mask、softmax、PV 可以解释融合 attention 的数学过程，但必须标记为“数学分解／未独立物化”，不能捏造独立运行时 OP、score tensor 或内存中间结果。
- selected KV 若只是 paged kernel 根据 block indices／block table 做逻辑寻址，就画元数据依赖和 paged K/V 消费关系；不要凭空增加 Map／Gather／KV Views 实际 OP。若所选实现确有物化 gather，就保留并给出证据。
- 核对残差、旁路、共享专家和 gated routing 的真实依赖。为了减少弯线不能删依赖或让融合节点重复承担同一计算。

## S3. 缓存、选择与 mask

- 主 K/V cache 与 index key-only cache 分别说明生产者、写入时机、布局、生命周期和消费者，按后端分别核验，不能因 Transformers 图里没画就断言所有实现无 index cache。
- slot_mapping、block_table、positions、position_ids、sequence lengths 等说明到底控制写入槽位、读页映射、RoPE 位置还是 causal 边界；不得用一个“runtime”框隐去不同作用。
- index scores、未来 key mask、block pooling、top-k 的顺序、归约轴、分组粒度、短序列边界、padding 和无效块处理来自实际代码，不预设所有模型都是同一种 block max。
- 每个 query／head group 的选择方式、index heads 与主 KV heads 的对应关系、K 与 V 使用同一选择的条件要明确。
- block selection mask、index 的未来约束和最终 token causal／pad mask 分层表示。候选 block 被选中不代表其中所有 token 都可见，Top-K 不是完整 causal mask 的替代品。
- 展示 attention mask 矩阵时核验行列轴、块大小、query group、被选／未选与 causal 不可见三者关系；实际数据示例可复算，示意数据明确标为示意，不能把不同 group 画成未经证明的同一 mask。
- prefill 与 decode 的 query/key 长度、缓存更新和 kernel 路径不同则分别标注，不能用一个 shape 同时冒充两者。

## S4. 公式与 shape

- 对每个公式检查矩阵方向、transpose、广播、reshape、head/group 轴、softmax 轴、缩放位置、mask 顺序、残差和 gating；乘法／加法各维度应成立。
- 所有 norm 明确具体变体。Gemma RMSNorm 与普通 RMSNorm 的参数约定不同；按当前实现核验 `1 + weight`、epsilon、归一化轴、精度和 cast。不要仅把节点改名而保留错误公式，也不要把所有新模型的 norm 写成 Gemma。
- RoPE 核验旋转维度、partial 比例、频率／scaling、位置广播，以及其与 norm/cache 的先后顺序。未旋转维度同样保留。
- MoE 核验 router dtype、score 激活、correction bias 用于选路还是权重、top-k、归一化、缩放、共享专家、专家激活与最终合并。区分总专家数和每 token 激活专家数。
- LaTeX 必须实际渲染无错误；解析器通过只是语法证据，数值样例／shape 检查辅助验证数学语义，不能替代读取代码。

## S5. 所有参数与符号

- 建立逐项符号表：符号、语义、轴／单位、适用阶段、配置键、具体值或表达式、证据、页面出现位置。所有可见符号包括上／下标和特殊记号都有定义。
- 图中优先用语义符号，参数参考中保留实际配置值；不能简单搜索替换数字。两个数值都为 128，并不代表同一概念：`D_h`（head dimension）、`D_idx`（index dimension）、`B_block`（tokens per block）应分别核验命名。
- 区分 `B`（batch）、`S`（query length）、`T`（key/cache length）、`N_h`、`N_kv`、`N_idx`、`H`、`E`、`K_block`（候选块预算）、`K_sel`（可见／逻辑候选 token 数）。沿图跟踪所有换轴，说明运行时 `[Nq,...]` 与参考 `[B,S,...]` 的关系。
- 数值事实如层数、总参数、激活参数、checkpoint 大小、shard 数和上下文上限保留有来源的值。标明精确值／近似值、字节单位与 dtype；不要将权重储存大小当作推理峰值显存。
- 权重 key、shape、dtype、参数量及 tensor-parallel 的逻辑／本地 shape 对照配置和 checkpoint。若未提供权重索引，应标注未验证，不能虚构 shard 文件。

## 输出要求

提交覆盖矩阵：每个节点／公式／符号／可见参数／后端路径映射到源码位置、核验方法与结果。不适用项附依据。发现差异给出对应代码和文档位置，区分“实现错误”“说明遗漏”“未提供证据”。未证明的核心参数或执行路径不允许 PASS。
