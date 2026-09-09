# Archify 必需依赖与页面集成

上游：[tt-a1i/archify](https://github.com/tt-a1i/archify)。它提供 typed JSON → HTML/SVG 的绘图与验证工作流；这是本技能必需的创作工具，不是 model-atlas 的 npm 运行时依赖。参考上游 [Skill](https://github.com/tt-a1i/archify/blob/main/archify/SKILL.md) 和 [交付契约](https://github.com/tt-a1i/archify/blob/main/archify/references/delivery-contract.md)，执行细节以实际安装版本为准。

## 依赖检查

1. 从技能目录发现当前 archify，读取其完整 SKILL.md，记录安装版本与可得的 revision；不能只在自己的说明中写“用了 archify”。
2. 在该技能根目录执行 `node bin/archify.mjs doctor`。找不到技能、缺少 Node 或渲染器不可用时报告绘图阻塞；可继续研究 docs，不静默降级为其他绘图工具或声称完成。
3. 缺失时向用户说明从 `tt-a1i/archify` 仓库的 `archify` 目录安装完整技能包；使用环境实际支持的安装机制。检查已有版本，不自动覆盖/升级，不把其源码或 node_modules 复制进 app。
4. 绘图主 agent 使用 archify；两个 reviewer 验证其源、回执与真实页面，不必重复生成同一冻结产物。若需要重生成，由主 agent 建立下一轮。

## 研究 → 绘图 → 接入

1. 先从模型成果 docs 中取得已追溯的节点、边、公式与符号映射。选符合语义的图类型：计算数据流可选 dataflow，整体模块图可选 architecture，执行顺序可选 workflow；不要为迁就某种样式错误标记模型对象的语义类型。
2. 按 archify 的 fast authoring path 读取一种匹配 schema、common schema 和一种示例，然后直接写候选 JSON；先有候选再查 renderer internals。使用 fresh stable ids，默认自动走线和 showcase profile。复杂模型拆为总览及逐层/逐模块子图，保留跨图映射和全部必需依赖，不为满足单图节点建议而删掉计算步骤。
3. 图示源保存于 `docs/models/<model-id>/diagrams/<diagram-id>.json`；自动验收产物和回执先生成到 `.scratch/add-<model-id>/artifacts/rNN/archify/`。每次修改源后 validate；只消费真实 diagnostics、subject、evidence 和 supportedFixes 修复。
4. 当前接口示例如下（在 archify 根目录运行，路径换成实际绝对路径）：

   ```text
   node bin/archify.mjs validate <type> <candidate.json> --quality showcase --json
   node bin/archify.mjs deliver <type> <candidate.json> <output.html> --quality showcase --json
   node bin/archify.mjs visual-check <output.html> --json
   ```

   根据实际版本读取完整质量检查回执；当前 showcase 契约要求 9 项检查、0 errors、0 warnings，不能将仅 4 项基础检查宣称为 showcase 通过。deliver 非零时目标可能仍是上一次成功产物，不能继续检查该旧文件冒充新候选。遵守 archify 自身的修复次数/停机限制；遇到其停止条件时记录阻塞，不通过新轮次编号绕过。
5. 成功 deliver 后只检查精确产物，不手改生成 HTML/SVG。visual-check 是自动浏览器证据，不是人工/图像 agent 的视觉结论；三个状态分别记录：确定性 validation/delivery、browser evidence、实际视觉 review。命令失败、未执行、检查通过互不冒充。
6. 将成功交付的 HTML 按字节原样复制到 `public/models/<model-id>/diagrams/<diagram-id>.html`，校验与回执 hash 一致。只复制选定交付文件，不将同目录日志、源码快照或 sidecar 全部复制进 public。保留第三方署名/许可信息。
7. 模型 View 提供真实可见的图示入口或隔离 iframe，用户从当前模型页面可打开/查看 Archify 产物；不能只在 scratch 生成图，页面却使用另一套未核验图。保留 React 模型注册、公式/源码详情和符号参考。资源地址遵循仓库 base path，不能硬编码 `/models/...` 破坏 GitHub Pages 子路径。
8. iframe 给出描述性 title、合适尺寸和可直接打开图示的入口；主题、选择节点与详情联动只有在实际接口支持且测试通过时才承诺，不假设 Archify 提供 React 回调。不要将完整文档直接注入 React DOM，或为接入强制改写交付字节。当前 MiniMax 手写图无需因新增技能而整体迁移。

## 两套验证范围

- 对独立 Archify HTML，执行安装版本的完整 viewport、主题和 first-screen 契约，保留真实截图与回执。
- 对接入后的 model-atlas 页面，另执行本技能 visual-review 清单：注册入口、详情、遮挡、padding、可读性、按需拖动、切模型/窗口变化等。独立 HTML 通过不能替代嵌入后的页面通过。
- React 外壳使用仓库公共 tokens；Archify 内部通过其 schema 支持的输入设置风格。样式隔离不足或上游缺少某种能力时显式说明，不能以修改生成 HTML 来伪称原交付回执仍有效。
- 语义 reviewer 核验成果 docs → Archify JSON → 生成图示 → 页面说明的一致性，防止抽象分层后丢失 cache、mask、分组或融合边界。

## 版本冻结

将 archify 版本、图示类型、specification/artifact hash 与字节数、质量回执、浏览器证据状态和视觉 review 状态写入候选文档，纳入文档契约的证据链。每次编辑 JSON 或重新生成产物都要刷新回执和双审；禁止修改上次成功 HTML 来绕过生成失败。
