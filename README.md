# 大件运输路径优化 (GA+SA)

这是一个基于遗传算法 (GA) 与模拟退火 (SA) 的 Flask Web 应用，旨在为大件运输场景提供考虑运输成本、时间成本和碳排放成本的综合路径优化方案。

## ✨ 功能亮点

- **Web 界面**：使用 Bootstrap 5 构建现代化、响应式的用户界面。
- **文件上传**：支持上传包含物流网络数据的 CSV 文件。
- **动态参数**：允许用户动态输入碳价和时间成本，以适应不同业务场景。
- **多维度优化**：综合考虑运输费用、时间成本和碳排放成本，找到总成本最低的路径。
- **结果可视化**：
    - 以饼图和堆叠柱状图清晰展示各类成本的占比和绝对值。
    - 优化结果（路径、成本明细）会直观地显示在页面上。
- **结果下载**：支持一键下载包含完整优化结果的 JSON 文件。
- **一键部署**：已配置好 `Procfile` 和 `gunicorn`，可直接部署在 [Railway.app](https://railway.app/) 等 PaaS 平台上。

## 🛠️ 技术栈

- **后端**: Flask, Gunicorn
- **数据处理**: Pandas, NumPy
- **算法**: 自定义遗传算法与模拟退火
- **图表生成**: Matplotlib
- **前端**: HTML, Bootstrap 5, JavaScript

## 🚀 本地运行指南

1.  **克隆仓库**
    ```bash
    git clone https://github.com/meanjey/yunshulmj.git
    cd yunshulmj
    ```

2.  **创建并激活虚拟环境**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **运行应用**
    ```bash
    python app.py
    ```
    应用将在 `http://127.0.0.1:5000` 上运行。

## ☁️ 部署到 Railway

本项目已为 Railway 部署进行了优化，只需几步即可上线：

1.  **将代码推送到 GitHub**。
2.  在 **[Railway.app](https://railway.app/)** 上使用 GitHub 账户登录。
3.  点击 **"New Project"** -> **"Deploy from GitHub repo"**。
4.  选择你的项目仓库，Railway 将自动读取 `Procfile` 并完成部署。

## 📁 项目结构

```
.
├── app.py                  # Flask 应用主文件
├── ga_solver.py            # 遗传算法与模拟退火核心逻辑
├── plot.py                 # Matplotlib 图表生成函数
├── evaluate.py             # 成本评估函数
├── Procfile                # Railway 部署配置文件
├── requirements.txt        # Python 依赖列表
├── templates/
│   └── index.html          # 前端页面模板
├── static/
│   └── figures/            # 保存生成的图表
├── data/                   # 存储上传的 CSV 文件
└── results/                # 存储输出的 JSON 结果
``` 