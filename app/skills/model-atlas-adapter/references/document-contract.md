# 文档交接与版本契约

主 agent 在开始研究时创建本契约的文档；检视者接手前读取当前候选版本。结构遵循仓库 `.scratch/<feature>/spec.md` 与单 ticket 文件的约定，不更改 AGENTS.md 来容纳本 skill。

## 成果文档与协作文档的界线

按用途分类，不按“是否由 agent 写”分类：两类文档都可能由 agent 生成，但读者只需要模型知识成果，协作记录用于跟踪本次适配过程。

| 类别 | 位置 | 内容与读者 | 页面/提交规则 |
| --- | --- | --- | --- |
| 模型知识成果 | `docs/models/<model-id>/` | 结构、公式、符号、源码解释、证据索引、图示源；面向模型读者与后续维护者 | 可随适配提交；页面消费其中已验证的模型内容 |
| 可执行适配代码 | `app/models/<model-id>/` | React/TS/CSS 等实际实现，不是 agent 交流文档 | 随适配测试和提交 |
| 生成图示 | `public/models/<model-id>/diagrams/` | 按 Archify 回执原样接入的 HTML 等用户可见产物 | 随页面验收；复制到 public 会进入静态构建，须先检查内容 |
| 协作与验证过程 | `.scratch/add-<model-id>/` | spec、任务、问题、交接、review、答复、轮次、日志和临时证据；面向三个 agent | 默认不纳入交付提交、不作为页面输入，不复制到 public/dist |

`docs/models/<model-id>/` **没有 agent 交互文档**。缺陷讨论、待派任务、review verdict、worker id、会话和修复过程全部留在 `.scratch/`；成果 docs 只保留凝练后的模型事实、公式、证据及真实限制。`.scratch` 在当前仓库并非自动被 git 忽略，提交时必须显式核对文件清单，避免 `git add .` 将过程记录带入 PR。用户要求审计材料入库时单独列明范围并检查敏感信息，不将其归入模型知识目录。

## 文件与写入所有权

```text
docs/models/<model-id>/
  README.md                      主 agent：成果导航、模型/后端版本与验证状态
  architecture.md                主 agent：模型结构、层族与数据流
  formulas.md                    主 agent：数学公式、推导、shape 和假设
  symbols.md                     主 agent：符号、参数与形状约定
  implementation.md              主 agent：源码调用链、融合边界和后端差异说明
  evidence.md                    主 agent：节点／公式／权重到固定源码的映射
  diagrams/<diagram-id>.json     主 agent：可复现的 Archify typed JSON 图示源
app/models/<model-id>/            主 agent：实际可执行页面适配代码
public/models/<model-id>/diagrams/
  <diagram-id>.html              主 agent：Archify 交付的用户可见图示
.scratch/add-<model-id>/
  spec.md                         主 agent：范围、来源、角色、阻塞和验收条件
  handoff.md                      主 agent：当前候选、待办、文档导航
  issues/NN-<slug>.md             主 agent：每个需修复问题单独一票
  candidates/rNN.md              主 agent：冻结的候选清单与内容指纹
  responses/rNN.md               主 agent：对检视 finding 的逐项答复
  reviews/visual-rNN.md           图示 agent 独占写入
  reviews/semantic-rNN.md         语义 agent 独占写入
  artifacts/rNN/                 主 agent 生成的截图、几何数据和测试日志
  artifacts/review-visual-rNN/    图示 agent 的补充证据
  artifacts/review-semantic-rNN/  语义 agent 的补充证据
  sources/                      主 agent：冻结的上游片段/元数据，仅供本次核验
```

目录已存在则读取并接续，避免覆盖旧轮次。主 agent 不改检视者的结论；检视者不改 app、测试、研究事实或对方报告。问题的 triage Status 采用仓库现有词汇；复查结果用独立的 `Review:` 字段，避免混入 triage 枚举。消息携带路径、轮次与通知；运行时要求的简短完成摘要（如三句话的 worker_done）必须正常提供，详细发现和决策仍以文档为准。

## 研究文档需要回答什么

- README：成果文件导航、模型变体、适用后端/阶段、固定来源、内容验证状态；它是模型成果索引，不复制 handoff 或任务列表。
- architecture：输入和输出，所选变体、模态和层族，算子拓扑，推理阶段与缓存，数学解释与实际执行的区分，已知限制。
- formulas：逐个算子的公式、定义域、轴、缩放、mask、广播和必要推导。用公式 id 链接符号表与源码证据，不能只把公式散落在 agent 讨论中。
- implementation：解释真实源码与公式/拓扑的对应关系，可附必要短代码片段和固定源链接；完整的可执行适配代码仍在 app，不把解释性代码块当运行实现。
- evidence：固定仓库 SHA 或粘贴文件 hash，节点／边 id、公式 id、配置／权重来源、类函数定位和页面对应位置；不同后端分栏。
- symbols：所有形状轴、公式符号和数值事实，维度变换与等式、配置值和来源。图上的简写链接回这里及页面说明。
- diagrams：按 Archify schema 保存真实源文件，稳定 node/edge id 映射到上述成果文档与页面；原始布局诊断、迭代截图和交付回执留在协作证据目录。

模型成果 Markdown 文件头显式标记 `Document-Kind: model-knowledge`、`Authoring: agent-generated, source-grounded`、`Model:`、`Source-Revisions:` 和 `Content-Status: draft | verified`。协作 Markdown 标记 `Document-Kind: agent-collaboration`、`Owner:`、`Round:`；已有 ticket 的 Status 约定保持不变。JSON 图示源遵守 Archify schema，不为分类硬塞不支持字段，其用途由成果 README 的清单说明。

协作结论提升为成果时，由主 agent 将核实的模型事实整理到对应成果文件并补源码引用，而不是拷贝对话或 PASS 报告。成果可链接固定源码和仓库代码，但不能依赖只存在本地的 `.scratch` 路径、临时绝对路径或私有日志才能读懂。只有同版双审完成后才允许 `Content-Status: verified`；该标识变更也属于新内容指纹，两位 reviewer 至少再次确认仅为状态提升且成果/产物没有其他变化，避免自我批准或拿旧 PASS 验收。

可用如下结构作为 evidence 表头，按需要细化，不要求把未知值填成猜测：

| 页面/节点/边/公式 ID | 后端与阶段 | 语义/shape | 源文件@SHA:符号 | 配置或权重证据 | 核验状态 |
| --- | --- | --- | --- | --- | --- |

## 候选快照

`rNN` 是评审轮次，不是代码身份。每次候选必须记录：

- 仓库路径、HEAD SHA、完整适配范围及上游固定版本。
- **内容指纹**：列出会影响构建的源码、样式、公共模块、测试、研究 docs、依赖锁文件、构建配置及必要静态资源，对文件字节逐项计算 SHA-256，并对有序的 `path + hash` 清单计算总 hash。纳入新增／未跟踪文件；相对上一轮被删除的文件记录 DELETE。Git HEAD 或 `git diff` 不能覆盖未提交／未跟踪文件，不能单独作为指纹。
- **证据指纹**：截图、测量文件和测试日志的路径及 hash。评审报告和交接文档不参与源码总 hash，避免评审写入导致循环失效；候选文件本身不哈希进自己。
- **上游证据**：保存实际使用的原始片段、配置、依赖代码和权重元数据的 hash、固定来源与获取时间。将这些小型来源快照纳入指纹范围，例如 `.scratch/add-<model-id>/sources/`；链接到 main 不能替代冻结证据。无需保存整套权重。
- **Archify 证据**：工具安装版本/可得 commit、图示 JSON 与 HTML 的 SHA-256、字节数、validate/deliver/visual-check 回执、以及 JSON id → 页面节点/章节的映射。图示源计入内容指纹，交付与浏览器回执放在 artifacts/rNN；再次生成会使旧回执和旧 review 失效。
- 生成时间、命令、服务 URL、构建／bundle 标识、视口、缩放、主题、展开状态、详情开关、覆盖场景及预期节点清单。
- 必须有足以回溯的具体文件清单，不仅写“工作区最新”。快照冻结后主 agent 不继续改这些文件，直到两个检视者完成本轮；可继续处理未纳入该快照的交接工作。

主 agent 发起检视前、两个检视者开始和结束时、主 agent 最终验收时各核对指纹。发现变化就报告 STALE 并建立新轮次，不能沿用旧 PASS。评审的补充截图放独占目录并注明同版构建，不能覆盖主 agent 截图。

### 可复现指纹工具

使用技能的 [../scripts/fingerprint.mjs](../scripts/fingerprint.mjs)，它只读取输入并向 stdout 输出 JSON，不修改文件：

```text
node <skill>/scripts/fingerprint.mjs snapshot <repo-root> app tests docs/models/<model-id> package.json package-lock.json tsconfig.json index.html public .scratch/add-<model-id>/sources
node <skill>/scripts/fingerprint.mjs verify <repo-root> <source-manifest.json>
```

路径列表只是示例，主 agent 和 reviewer 应核对当前构建配置、依赖闭包与实际范围，补充遗漏的入口／配置／资源；工具不能推断未声明的构建依赖。主 agent 将 snapshot 的 JSON 原样保存到候选目录，并在 rNN.md 链接它。对冻结的 artifacts/rNN 另做一次 snapshot，保存证据 manifest；manifest 均放在所哈希范围之外。

工具将路径规范化为仓库相对 `/` 路径，以确定性字符排序列举全部文件（包含未跟踪文件）；每个文件按字节做 SHA-256。总 hash 使用 UTF-8 的紧凑 JSON `{schema,scopes,files}`，hash 均为小写。缺失输入记录 null，目录成员变化会改变清单；verify 会重新枚举 scope，所以新增文件不会因旧清单未列出而漏检。拒绝越出 root 或符号链接，遇到此类输入先显式提供可验证的普通文件快照。工具报错或校验不一致不可 PASS。

## 检视报告模板

```markdown
# Visual / Semantic review — rNN

Reviewer: 实际 agent / dispatch 标识
Candidate: 候选文件路径
Source fingerprint: 实测值
Evidence fingerprint: 实测值
Started / Finished: 时间
Verdict: PASS | CHANGES_REQUIRED | BLOCKED | STALE

## Coverage

逐项列出已检视场景或节点／公式／符号映射，及有依据的 N/A。

## Findings

### V-001 / S-001 — 简短标题

Rule: V2 / S4 等
Severity: blocker | defect | suggestion
Location: 文件位置、节点／边／公式 id
Evidence: 截图、几何数据或固定源码引用
Expected / Actual: 可复查的差异
Requested change: 需要满足的约束，不越权修改实现
Recheck: 新轮次的结论与证据

## Cross-review questions

给另一角色的问题和对应文档位置。

## Remaining limitations

未覆盖项、未知事实、工具阻塞；有关键未验证项时不可 PASS。
```

`PASS` 仅表示该角色的全部适用必检项已通过；suggestion 可作为有说明的非阻塞建议。范围内 blocker 和 defect 必须解决并被检视者复查关闭，不能由主 agent 降级以绕过验收。

混合情况的结论优先级：候选变化为 STALE；候选稳定但关键检视因缺失证据／工具而无法执行为 BLOCKED；可完整检视但存在已确认缺陷为 CHANGES_REQUIRED；其余才可能 PASS。无论最终 verdict 是什么，都保留已发现的问题。NOT-RUN 仅是单项覆盖状态，不是新的报告 verdict。

## 主 agent 答复与收敛

responses 表逐项记录 finding id、关联 issue、修复文件／约束、增加的测试、新证据、是否请求重新检视。结论相冲突时，把问题和证据送回对应角色；优先以固定源码决定语义，在保持语义的前提下调整布局。

如果同一问题连续两轮重现，记录根因诊断和替换的约束，而非追加又一条覆盖 CSS。如果需要新来源、权限或用户选择，写出具体阻塞和所需输入；保留已完成材料，但不得把三 agent 流程记作完成。
