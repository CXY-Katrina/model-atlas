---
name: model-atlas-adapter
description: 基于用户提供的 vLLM 或 Transformers 源码，为 CXY-Katrina/model-atlas 新增或完善模型结构介绍、公式、权重证据和交互图示，绘图必须使用 tt-a1i/archify。主 agent 负责研究、文档和实现，自动启动图示与语义两个独立检视 agent，以文档交接并迭代验收。用于该仓库的模型适配，不用于泛泛讲解模型或仅绘制一张示意图。
---

# Model Atlas 模型适配

将可追溯的模型源码转为研究文档和仓库原生的交互式结构介绍。采用三个角色：主 agent 是唯一实现者；两个新检视 agent 分别负责视觉布局和公式／源码／参数语义。结果必须有真实渲染和同版双审证据。

## 必需绘图依赖

使用 [tt-a1i/archify](https://github.com/tt-a1i/archify) 技能生成 typed JSON → HTML/SVG 图示，不是仅借鉴配色或布局。主 agent 先完整读取当前安装的 archify `SKILL.md`，按 [references/archify-integration.md](references/archify-integration.md) 检查依赖、生成、验证并接入页面。缺少 archify 或其渲染器无法运行时可以继续模型研究，但绘图阶段阻塞；不得静默替换为手写图并声称已满足依赖。

## 三类产物

- `docs/models/<model-id>/`：主 agent 梳理的模型知识成果，包括结构、公式、符号、源码解释和 Archify 图示源。面向读者，不放 agent 对话、任务分配或审查过程。
- `app/models/<model-id>/`：实际可执行的模型页面适配代码；验收后的 Archify 静态图示按集成约定放到 `public/models/<model-id>/diagrams/`。
- `.scratch/add-<model-id>/`：三个 agent 的协作记录、review、答复、候选版本、临时截图与日志。它们不是模型介绍，不直接渲染或作为网站资源发布。

详细文件分类、标识、所有权和提升为正式成果的规则见 [文档契约](references/document-contract.md)。

## 开始前

1. 确认工作区是用户指定的 `CXY-Katrina/model-atlas` 仓库；读取实际 `AGENTS.md`、它指向的仓库约定、`docs/model-extension.md`、注册接口及现有测试。存在相关领域文档时遵循其术语。仓库路径、默认分支和启动命令从当前环境发现，不硬编码旧会话路径或端口。
2. 接受文件、粘贴代码、仓库 URL、commit/tag 或模型目录。明确模型 id，建立变体／模态／后端／推理阶段的范围矩阵。只有一个后端时即可研究，另一个后端标记“未提供／未验证”；缺失内容不能靠另一模型补齐。关键未知项影响必需拓扑、公式或参数真伪，可选未知项不影响约定介绍；逐项说明分类依据。缩减用户要求的覆盖范围须先取得用户同意，不能删掉必检节点以获得 PASS。
3. 对所有用户源码保留内容哈希、原始 ref 和获取时间。URL 指向浮动分支时解析固定 SHA；若用户提供的是之前保存的文件，先比对其字节与该 revision，不能把今天的 main SHA 冒充原文件出处。各依赖和后端分别固定版本；不匹配时记录差异或保留为来源未确定的哈希片段。提供的片段不能证明完整架构时追踪同版本源码；关键事实仍无法确定则询问用户。通常读取源码、配置和权重索引／header 即可，无需下载全部权重或运行上游模型。
4. 告知用户采用此 skill 和三 agent 文档闭环。适配授权包括本地文档、代码和测试，不默认包含推送、发布 Pages、公开上传截图或修改上游实现。

## 工作流

### A. 建立证据与文档

主 agent 先读 [references/semantic-review.md](references/semantic-review.md)，按 [references/document-contract.md](references/document-contract.md) 创建研究文档、范围与待核实事项。

研究覆盖输入／embedding、各类层、attention、FFN／MoE、缓存与输出；视觉或其他模态按实际模型补充。将每个可见节点、公式、shape、参数事实绑定到固定源码证据，区分数学解释与实际运行步骤。

完成条件：所选范围的节点／边清单、公式、符号表、后端差异和证据索引已写入成果 docs；任务、问题和答复写入协作目录。未知项已显式记录，未被当作事实渲染。

### B. 适配与首次渲染

主 agent 读 [references/implementation.md](references/implementation.md)、[references/visual-review.md](references/visual-review.md) 和 [Archify 集成约定](references/archify-integration.md)，用 archify 生成并验证图示，再实现独立模型模块、注册入口和可见研究内容。架构图使用实际生成的交互 HTML/SVG，不用栅格图或另写一套布局替代 Archify 产物。

根据当前 scripts 启动或复用本地服务，先验证服务对应当前工作区与代码。运行类型／构建／测试，实际打开页面，覆盖概览、所有不同结构的展开图和详情页。记录 URL、截图、视口、主题及运行日志；不能仅凭构建成功声称已渲染。

完成条件：生成一个可复查的候选版本及渲染清单，按文档契约冻结其源文件与证据指纹。

### C. 自动启动两个检视 agent

在候选版本就绪后，读取 [references/collaboration.md](references/collaboration.md)，启动两个独立的新 agent 并行检视。主 agent 负责所有代码修改；检视 agent 只写各自 review 文档。首轮不得复用主 agent 的身份充当检视者。

- 图示 agent：阅读视觉清单、候选文档，独立检查实际页面及所有相关截图。
- 语义 agent：阅读语义清单，独立对照固定源码、公式、参数与页面展示。

为每个 agent 明确工作区、候选指纹、可写文档、引用资源、输出格式和禁止执行的外部动作。运行时消息只通知“哪个文档的哪一版已更新”，结论与修复答复写入文档。

### D. 修复与复查

主 agent 汇总两份报告，逐项修复或提供证据答复。每次修改模型实现、共享样式／路由、研究事实或截图后生成新候选版本；旧版本 PASS 自动失效。重新渲染受影响场景，两位检视者对最新同版分别复查。

同一缺陷连续两轮重现时检查布局约束或语义根因，停止堆叠局部偏移补丁。超时不代表检视通过；外部资源、权限或关键模型事实确实缺失时记录阻塞并向用户请求必要信息。不得为凑齐验收而伪造 agent 或 PASS。

### E. 验收与交付

只有以下条件同时满足，才能称“适配完成”：

- 文档与实际页面内容一致，全部可见节点、公式、符号、参数均有核验记录；无未解决的范围内缺陷或关键未知项。
- Archify 的源、产物及 validation／delivery／browser 回执可追溯，页面接入的字节与交付回执一致；实际视觉双审与工具检查分别通过，未泄露协作记录。
- 类型检查、构建、适用测试和真实浏览器检查通过；共享改动也回归已有模型。
- 两份 review 对同一候选指纹、同一源码版本分别 PASS，最终验证期间文件未变化。
- 本地访问地址与截图可定位，已记录限制和有依据的不适用项。

最终说明模型范围、文件位置、本地访问方式、测试与双审结果。用户明确要求提交／部署时才执行：确认目标分支与文件范围，排除无关文件，推送后跟踪对应提交的 Pages 工作流，并验证线上入口及资源版本。未部署时明确说明是本地版本。
