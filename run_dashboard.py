#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启动回测结果可视化仪表盘的入口脚本
"""

import argparse
from src.visualization.web_dashboard import run_server
from src.utils.logger import get_logger

logger = get_logger("Dashboard")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动HKStock-Quant回测结果可视化仪表盘')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='服务器主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8050,
                        help='服务器端口 (默认: 8050)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info(f"启动回测结果可视化仪表盘，访问地址: http://{args.host}:{args.port}/")
    logger.info("按 Ctrl+C 停止服务器")
    
    try:
        run_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("服务器已停止")

if __name__ == "__main__":
    main()
