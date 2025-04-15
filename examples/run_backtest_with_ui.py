#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行回测并自动启动UI界面展示结果的示例脚本
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import BacktestEngine
from src.strategies.moving_average import MovingAverageStrategy
from src.utils.logger import get_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行回测并启动UI界面')
    parser.add_argument('--symbols', type=str, nargs='+', default=['0700.HK', '9988.HK', '0388.HK'],
                        help='要回测的股票代码列表')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                        help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-01-01',
                        help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='初始资金')
    parser.add_argument('--short_window', type=int, default=20,
                        help='短期移动平均窗口')
    parser.add_argument('--long_window', type=int, default=50,
                        help='长期移动平均窗口')
    parser.add_argument('--ma_type', type=str, default='EMA', choices=['SMA', 'EMA'],
                        help='移动平均类型 (SMA 或 EMA)')
    parser.add_argument('--port', type=int, default=8050,
                        help='UI界面服务器端口')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 初始化日志
    logger = get_logger("BacktestWithUI")
    logger.info("启动回测并展示UI界面")
    
    # 定义回测参数
    symbols = args.symbols
    start_date = args.start_date
    end_date = args.end_date
    initial_capital = args.capital
    
    # 定义策略参数
    strategy_params = {
        'short_window': args.short_window,
        'long_window': args.long_window,
        'ma_type': args.ma_type
    }
    
    # 初始化回测引擎
    logger.info(f"初始化回测引擎，股票: {symbols}, 时间段: {start_date} 至 {end_date}")
    engine = BacktestEngine(
        strategy=MovingAverageStrategy,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=initial_capital,
        data_source='auto'
    )
    
    # 运行回测
    logger.info("运行回测...")
    results = engine.run(strategy_params=strategy_params)
    
    # 打印结果
    logger.info("回测完成")
    logger.info(f"总收益率: {results['metrics']['total_return']:.2%}")
    logger.info(f"年化收益率: {results['metrics']['annual_return']:.2%}")
    logger.info(f"夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"最大回撤: {results['metrics']['max_drawdown']:.2%}")
    logger.info(f"胜率: {results['metrics']['win_rate']:.2%}")
    logger.info(f"盈亏比: {results['metrics']['profit_factor']:.2f}")
    logger.info(f"交易次数: {results['metrics']['total_trades']}")
    
    # 保存回测结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = engine.save_results(filename=f"backtest_MovingAverageStrategy_{timestamp}.json")
    logger.info(f"结果已保存至: {results_path}")
    
    # 启动UI界面
    logger.info(f"启动UI界面，端口: {args.port}")
    dashboard_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_dashboard.py")
    
    try:
        # 使用子进程启动仪表盘
        dashboard_process = subprocess.Popen(
            [sys.executable, dashboard_script, "--port", str(args.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待服务器启动
        time.sleep(2)
        
        # 检查进程是否正常运行
        if dashboard_process.poll() is None:
            logger.info(f"UI界面已启动，请在浏览器中访问: http://127.0.0.1:{args.port}/")
            logger.info("按 Ctrl+C 停止服务器")
            
            # 等待用户中断
            try:
                dashboard_process.wait()
            except KeyboardInterrupt:
                logger.info("正在关闭服务器...")
                dashboard_process.terminate()
                dashboard_process.wait()
                logger.info("服务器已关闭")
        else:
            stdout, stderr = dashboard_process.communicate()
            logger.error(f"启动UI界面失败: {stderr.decode('utf-8')}")
            
    except Exception as e:
        logger.error(f"启动UI界面时出错: {e}")
    
    return results

if __name__ == "__main__":
    main()
