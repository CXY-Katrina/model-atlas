# Model Atlas

MiniMax-M3 模型结构与算子连线的纯静态交互页面。项目使用 Vite、React 和 KaTeX，可直接部署到 GitHub Pages，不依赖服务器、数据库或 Cloudflare Workers。

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
