#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 uv 初始化项目环境的脚本
此脚本会检查 uv 是否已安装，如果没有则安装，然后创建虚拟环境并安装项目依赖
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_uv_installed():
    """检查 uv 是否已安装"""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        print("✓ uv 已安装")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ uv 未安装")
        return False

def install_uv():
    """安装 uv"""
    print("正在安装 uv...")
    
    try:
        # 使用官方推荐的安装方式
        if platform.system() == "Windows":
            # Windows 安装命令
            subprocess.run(
                ["curl", "-sSf", "https://raw.githubusercontent.com/astral-sh/uv/main/install.ps1", "|", "powershell", "-"],
                check=True,
                shell=True
            )
        else:
            # macOS/Linux 安装命令
            subprocess.run(
                ["curl", "-sSf", "https://raw.githubusercontent.com/astral-sh/uv/main/install.sh", "|", "bash"],
                check=True,
                shell=True
            )
        
        print("✓ uv 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ uv 安装失败: {e}")
        print("请手动安装 uv: https://github.com/astral-sh/uv#installation")
        return False

def create_venv():
    """创建虚拟环境"""
    print("正在创建虚拟环境...")
    
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    
    try:
        subprocess.run(["uv", "venv", ".venv"], check=True)
        print(f"✓ 虚拟环境创建成功: {venv_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 虚拟环境创建失败: {e}")
        return False

def install_dependencies():
    """安装项目依赖"""
    print("正在安装项目依赖...")
    
    try:
        # 安装主要依赖
        subprocess.run(["uv", "pip", "install", "-e", "."], check=True)
        
        # 安装开发依赖
        subprocess.run(["uv", "pip", "install", "-e", ".[dev]"], check=True)
        
        print("✓ 依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("正在创建必要的目录...")
    
    directories = [
        "data/raw",
        "data/processed",
        "logs",
        "logs/visualizations",
        "logs/reports",
        "notebooks"
    ]
    
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {directory}")
    
    return True

def create_env_file():
    """从 .env.example 创建 .env 文件"""
    print("正在创建 .env 文件...")
    
    env_example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if os.path.exists(env_path):
        overwrite = input(".env 文件已存在。是否覆盖？(y/n): ")
        if overwrite.lower() != 'y':
            print("跳过 .env 文件创建。")
            return True
    
    try:
        import shutil
        shutil.copy(env_example_path, env_path)
        print(".env 文件创建成功。")
        print("请编辑 .env 文件设置您的 API 密钥和其他配置。")
        return True
    except Exception as e:
        print(f"创建 .env 文件时出错: {e}")
        return False

def activate_venv_instructions():
    """显示激活虚拟环境的指令"""
    print("\n要激活虚拟环境，请运行:")
    
    if platform.system() == "Windows":
        print(".venv\\Scripts\\activate")
    else:
        print("source .venv/bin/activate")

def main():
    """主函数"""
    print("=" * 50)
    print("HKStock-Quant 项目初始化 (使用 uv)")
    print("=" * 50)
    
    # 检查 uv 是否已安装
    if not check_uv_installed():
        if not install_uv():
            return
    
    # 创建虚拟环境
    if not create_venv():
        return
    
    # 安装依赖
    if not install_dependencies():
        return
    
    # 创建目录
    create_directories()
    
    # 创建 .env 文件
    create_env_file()
    
    print("\n" + "=" * 50)
    print("初始化成功完成！")
    print("=" * 50)
    
    # 显示激活虚拟环境的指令
    activate_venv_instructions()
    
    print("\n下一步:")
    print("1. 编辑 .env 文件设置您的 API 密钥和其他配置")
    print("2. 运行回测示例: python examples/backtest_example.py")
    print("3. 探索 notebooks 目录中的 Jupyter 笔记本")
    print("4. 如需实盘交易，请在 src/config/futu_config.yaml 中配置 Futu API")
    print("\n祝您交易愉快！")

if __name__ == "__main__":
    main()
