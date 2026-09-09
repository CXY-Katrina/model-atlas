# 模型扩展与公共样式

## 目录职责

```text
app/
  page.tsx                 公共页面：选择模型、主题、链接、帮助与状态重置
  graph/                   公共画布、节点、拖动与路由算法
  styles/tokens.css        颜色、间距、圆角、图示公共尺寸
  styles/ui.css            公共页面与节点样式
  models/
    registry.ts            ModelDefinition 接口与注册校验
    index.ts               已支持模型列表
    types.ts               张量权重、算子、源码证据的公共类型
    minimax-m3/
      index.ts             元信息、资料链接、View 与 Reference
      view.tsx             模型内部的层类型选择与详情状态
      diagrams.tsx         模型拓扑、端口、分支与合流约束
      operators.ts         Dense/Sparse 算子定义
      data.ts              checkpoint 权重与基础数据
      evidence.ts          源码片段及输入输出来源
      formulas.ts          公式及符号说明
      config.ts            官方配置值与符号表
      sources.ts           源码地址与固定提交
      details.tsx          MiniMax 详情展示及 shape 符号化
      reference.tsx        参数参考弹窗
      layout.css           仅适用于该模型的布局
```

公共画布不包含 MiniMax 的 head 数、层数、权重路径或算子名称。不同模型不必采用相同的 Dense/Sparse 分类，也不必复用 MiniMax 的符号替换规则。

## 新增模型

1. 在 `app/models/<model-id>/` 新建模型目录，提供符合 `ModelDefinition` 的导出。
2. 提供唯一、稳定的小写连字符 `id`，以及 `name`、`resources`、`facts`、`View`。`Reference` 可选。
3. `View` 管理该模型自己的图示、层选择和算子详情；展开或收起全屏图示时调用 `onExpandedChange(boolean)`。可复用 `graph/`，也可实现不同架构的展示。
4. 在模型入口导入自己的 CSS。布局选择器使用 `:where([data-model="<model-id>"])` 限定作用域，避免影响其他模型。若选择器起始元素就是 `.atlas-app`，使用 `.atlas-app:where([data-model="<model-id>"])`。
5. 在 `app/models/index.ts` 的 `createModelRegistry([...])` 中注册模型。不需要修改 `page.tsx` 或其他模型。

注册列表不能为空，重复或非法 id 会报错。未知 URL 参数回退到第一个已注册模型。

页面通过 `?model=<model-id>` 保存选择，支持浏览器前进/后退。切换模型会重建模型视图、清空展开与帮助状态，同时保留明暗主题。正式列表只包含有真实数据的模型；测试模型不参与构建入口。

## 公共变量

颜色统一在 `styles/tokens.css` 中声明；`.atlas-app.dark` 只覆盖主题差异。样式中使用语义变量，如 `--operator-border`、`--weight-surface`、`--edge-data`、`--focus-ring`。

常用几何变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `--graph-node-padding-inline` | `--space-18` | 节点文字左右留白 |
| `--graph-group-padding-inline` | `--space-24` | 背景分组左右留白 |
| `--graph-row-gap` | `--space-24` | 默认图示行间距 |
| `--graph-node-max-width` | `520px` | 节点最大宽度 |
| `--space-*` / `--radius-*` | 具名尺寸刻度 | 公共间距与圆角 |

模型可在自己的作用域覆盖公共变量。拓扑特有的网格宽度、行高和端口位置仍由模型布局定义，不应把所有图强制套成同一网格。

`graph/routing.ts` 的 `GRAPH_GEOMETRY` 集中定义端口比例、转角半径、避障距离、直线容差和拖动留白。单条边的 `approach` / `departure` 是该拓扑的布线选择，保留在模型图示中。

## 验证

- `npm test`：类型检查、生产构建、源码证据、路由、模型注册、独立模型渲染与公共样式约束。
- 先启动 `npm run dev -- --host 127.0.0.1 --port 5173`，再运行 `npm run test:browser`：真实浏览器检查三种宽度下的对齐/留白/文字边界，以及 Dense/Sparse、FFN、明暗主题、模型切换和历史导航。
- 浏览器检查默认使用 Windows Edge；其他环境通过 `BROWSER_PATH` 指定 Chromium 可执行文件，通过 `TEST_URL` 指定开发服务。检查截图写入系统临时目录。

保留模型的数值配置与 checkpoint 证据；仅图示展示使用符号，不能把具体模型的数值事实替换成全局常量。
