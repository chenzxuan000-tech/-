#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo ">>> 创建虚拟环境..."
    python3 -m venv .venv
fi

echo ">>> 激活虚拟环境..."
source .venv/bin/activate

if [ ! -f ".venv/.installed" ]; then
    echo ">>> 安装依赖..."
    pip install -r requirements.txt
    touch .venv/.installed
fi

echo ">>> 启动亚马逊广告诊断工具..."
echo "    浏览器打开 http://localhost:8501"
echo ""
streamlit run app.py --server.headless true --browser.gatherUsageStats false
