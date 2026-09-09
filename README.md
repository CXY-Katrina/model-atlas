# Model Atlas

支持按模块扩展模型的纯静态架构交互页面，当前提供 MiniMax-M3 的算子图与源码证据。项目使用 Vite、React 和 KaTeX，可直接部署到 GitHub Pages，不依赖服务器、数据库或 Cloudflare Workers。

## 本地开发

要求 Node.js `>=22.13.0`。

```bash
npm install
npm run dev
```

常用命令：

- `npm run check`：TypeScript 类型检查
- `npm run build`：生成 `dist/` 静态文件
- `npm test`：构建并运行全部测试
- `npm run preview`：本地预览生产构建
- `npm run test:browser`：开发服务启动后，执行图示布局、主题与模型切换的浏览器回归
- `npm run test:skill`：运行模型适配技能的候选指纹工具测试（也包含在 `npm test` 中）

## 代码结构与模型扩展

公共画布位于 `app/graph/`，颜色、间距、字号和圆角变量位于 `app/styles/tokens.css`。每个模型的图示、配置和源码证据独立存放在 `app/models/<model-id>/`。

新增模型实现 `ModelDefinition` 并注册到 `app/models/index.ts` 即可。完整说明见 [模型扩展与公共样式](docs/model-extension.md)。

## 使用 Skill 新增模型

仓库提供 [model-atlas-adapter](app/skills/model-atlas-adapter/SKILL.md)：基于用户提供的 **vLLM 或 Transformers 源码**，研究模型结构、生成文档，并适配为本项目的交互图示。技能包位于 `app/skills/model-atlas-adapter/`，不由前端导入，也不会作为页面运行时执行。

### 安装与调用

在支持技能安装的 Codex 环境中，可以发送：

```text
使用 $skill-installer，从 CXY-Katrina/model-atlas 仓库安装
app/skills/model-atlas-adapter 目录中的技能。
```

安装默认读取仓库 `main`。若试用尚未合并的 PR，请明确指定该 PR 的源分支或提交 SHA。也可以将该目录完整复制到当前环境的 `$CODEX_HOME/skills/model-atlas-adapter/`；未设置 `CODEX_HOME` 时使用 `~/.codex/skills/model-atlas-adapter/`。保留其中的 `references/`、`scripts/` 和 `agents/`，已有同名技能时先检查差异，避免覆盖本地定制。安装后在下一轮对话中调用。

在本仓库工作区发送：

```text
使用 $model-atlas-adapter，为 model-atlas 新增 <模型名称> 的结构介绍。

模型 id：<小写连字符 id>
源码：<vLLM 或 Transformers URL、仓库文件路径或粘贴代码>
版本：<commit/tag；未指定时先解析并固定源码版本>
范围：<模型变体、模态，以及需要展示的后端和推理阶段>

请研究源码、生成 docs、实现并渲染页面，
启动图示与语义两个独立检视 agent，通过文档修复并复查。
本次只完成本地适配，不提交或部署。
```

只提供一种后端也可以适配；未提供的实现不会被当作已验证事实。需要提交、创建 PR 或部署时，请另行明确目标分支和动作。

### 三 agent 文档闭环

| 角色 | 职责 |
| --- | --- |
| 主 agent | 学习源码、维护研究文档、实现模型模块、渲染页面并落实修复 |
| 图示 agent | 独立检查框线／文字遮挡、走线、对称、箭头、padding、溢出与画布交互 |
| 语义 agent | 独立核对图示、公式、源码、缓存、mask、权重及所有可见参数和符号 |

研究结果写入 `docs/models/<model-id>/`；范围、候选版本、问题、答复、双审报告和截图证据写入 `.scratch/add-<model-id>/`。主 agent 是唯一实现者，检视者各写自己的报告；三方通过这些文档交接和迭代。

验收要求真实浏览器检查和适用测试通过，且两个检视者对**同一候选版本**分别通过。源码或截图变化后，旧版验收失效；技能附带的 SHA-256 指纹工具会重新枚举文件，检测新增、修改和删除。布局约束包含直线优先、同类合流对称、内容自适应尺寸和统一留白；语义约束要求固定源码版本、区分数学分解与实际 OP，并按含义定义符号，不能把相同数字机械替换成同一维度。

完整流程分别见 [图示检查](app/skills/model-atlas-adapter/references/visual-review.md)、[语义检查](app/skills/model-atlas-adapter/references/semantic-review.md) 和 [文档交接契约](app/skills/model-atlas-adapter/references/document-contract.md)。

需要可用的多 agent 和浏览器能力：提供 Orca orchestration 的环境按其实际运行协议调度；其他支持原生子 agent 的环境使用对应机制。没有独立检视或真实渲染能力时会报告阻塞，不能用主 agent 自检冒充双审完成。

## GitHub Pages 部署

仓库包含 [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)。推送到 `main` 后，GitHub Actions 会自动构建并发布 `dist/`。

首次使用时，在 GitHub 仓库中打开：

1. **Settings → Pages**
2. 将 **Build and deployment → Source** 设置为 **GitHub Actions**
3. 推送 `main`，或在 **Actions → Deploy GitHub Pages** 中手动运行

Vite 使用相对资源路径，因此项目页（`https://<user>.github.io/<repo>/`）和自定义域名均可使用同一份构建产物。

## 运行时依赖

- React / ReactDOM：交互界面
- KaTeX：公式渲染

其余工具仅用于本地类型检查和静态构建。
