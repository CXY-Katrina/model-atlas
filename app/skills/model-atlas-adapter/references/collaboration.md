# 三 agent 文档协作

这是主 agent 监督并落实修复的协作任务，不是把任务完全移交给另一个 agent。角色固定为主 agent、图示检视者、语义检视者；检视者不再递归启动子 agent。

## 选择真实运行机制

- 环境提供 Orca orchestration skill 时，主 agent 必须先完整读取该 skill 和其当前 CLI 提供的版本化指南，按实际指南创建或绑定 Run、创建两个独立 Task，再各自启动 fresh worker。使用当前工作区，两个 reviewer 只写独占文档，通常不需要新 worktree。
- 不在本 skill 中固定 Orca 命令参数，避免 CLI 升级失效。遵守实际 dispatch、completion、ack 和 worker 清理规则；不以聊天消息代替已启动的 worker。
- 环境没有 Orca、但有原生子 agent 工具时，可用该环境支持的创建／发消息／等待机制，记录真实 agent id。若用户或环境要求 Orca 而其不可用，报告阻塞；不得静默换机制伪称 Orca 成功。
- 没有可用多 agent 能力时，可以继续研究和实现，但明确标记“独立双审未执行”，请求可用环境；主 agent 的两次自检不能冒充两个 agent。

## 首轮派发

先完成可评审候选，冻结源码、Archify 图示源/交付产物与主证据，再同时启动两个 reviewer。分配各自独立的浏览器页面和文档路径；不要让两人运行会修改同一个 fixture、截图路径或应用状态的脚本。主 agent 在评审期间保持候选文件稳定，整理交接和回复非修改性问题。reviewer 写入位置只在 `.scratch/add-<model-id>/` 的各自目录；`docs/models/<model-id>/` 是模型知识成果，不是三方留言板。

传递完整绝对路径或可解析的仓库相对路径，不能依赖 reviewer 恰好继承聊天上下文。使用以下任务内容，填写实际值后派发。

两位 reviewer 都必须读取 [Archify 集成约定](archify-integration.md)：图示角色检查真实交付字节、回执、独立 HTML 与嵌入页面；语义角色核验模型知识成果、typed JSON、生成图示和页面解释的映射。不要让 reviewer 直接重生成冻结图示。

### 图示 reviewer 任务

> 你是 model-atlas 本轮独立图示检视者，不是实现者。工作区为〔路径〕。先读取仓库 AGENTS 约定、技能中的 visual-review.md 和 document-contract.md，然后阅读〔candidate 路径〕、handoff、研究 docs 与渲染清单。独立打开〔URL〕并检查全部约定场景，读取实际截图与几何数据。关注框线／文字遮挡、连线绕行、端口与合流对称、短箭头、共享线粗细、分组归属、padding、画布适配以及交互后退化。仅写〔visual-rNN.md〕和〔review-visual-rNN 证据目录〕，不得修改源码、测试、其他报告或发布。开工和结束核对候选指纹，发现变化报告 STALE。输出覆盖、可重现 findings 和明确 verdict；关键未检项不得 PASS。需要语义解释时先写报告问题，通过协调器通知，不凭外观改语义。不要启动其他 agent。完成后按运行时的真实生命周期协议通知主 agent，消息只摘要并链接报告。

### 语义 reviewer 任务

> 你是 model-atlas 本轮独立语义检视者，不是实现者。工作区为〔路径〕。先读取仓库 AGENTS 约定、技能中的 semantic-review.md 和 document-contract.md，再读取原始用户源码〔固定版本／路径〕、〔candidate 路径〕和研究 docs。独立追踪依赖，核对所有可见节点、边、公式、shape、参数、权重与符号，包括页面概览、详情、帮助内容及后端差异；不要仅认可主 agent 摘要。实际查看渲染内容与公式。仅写〔semantic-rNN.md〕和〔review-semantic-rNN 证据目录〕，不得改源码、测试、其他报告或发布。开工和结束核对指纹，变化报告 STALE。输出逐项覆盖与 evidence、findings 和 verdict；未提供来源明确标注，关键未验证项不得 PASS。不要启动其他 agent。完成后按真实生命周期协议通知主 agent，消息链接报告。

## 轮次协调

1. 保存运行机制返回的两个身份／dispatch 与对应文档路径到 handoff；只有启动成功才称“两个 agent 已启动”。
2. 使用机制提供的等待事件，不忙轮询或无限 sleep。等待时保持用户进度更新；超时只是检查点，不是失败或通过。只在有证据证明 worker 失败时按当前机制恢复。
3. reviewer 先写自己的报告问题；主 agent 将问题路径通知另一角色，另一角色在自己报告答复。主 agent 写入 responses 汇总，避免同文件并发覆盖。
4. 收到两份报告后，主 agent 修复，刷新 docs、测试和截图，生成下一候选。任何实现改动均需要两角色对新指纹重新确认；可以重点复核影响范围，但必须说明未改区域的覆盖依据。
5. 后续轮次可用原 reviewer 的正式 follow-up／新 dispatch；已关闭则创建同角色的新 reviewer，给它完整文档。没有有效新复查报告时旧 PASS 不生效。
6. 对已结束的 worker 按实际运行机制释放或立即复用。不要因长时间没有消息就关闭仍活跃的 worker，也不保留无后续任务的空闲 terminal。
7. 主 agent 核对同版双 PASS、无未解决缺陷、完整测试与证据后写最终 handoff。若受阻，只交付已完成部分及明确阻塞，不承诺不存在的审查或部署。
