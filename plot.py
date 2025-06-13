import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def plot_results(result):
    """
    根据优化结果生成成本对比图
    
    Args:
        result: Result对象，包含path和costs属性
    """
    # 准备数据
    costs = result.costs
    labels = ['运输费用', '时间成本', '碳排放成本']
    values = [costs.freight, costs.time, costs.carbon]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # 创建饼图
    plt.figure(figsize=(10, 6))
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('成本构成分析')
    
    # 添加总成本信息
    plt.figtext(0.5, 0.01, f'总成本: {costs.total:.2f}元', ha='center', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    return plt.gcf()

def plot_cost_breakdown(result):
    """
    根据优化结果生成成本构成堆叠柱状图
    """
    costs = result.costs
    labels = ['总成本构成']
    freight_costs = [costs.freight]
    time_costs = [costs.time]
    carbon_costs = [costs.carbon]

    # 设置图形大小
    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制堆叠柱状图
    bar_width = 0.5
    ax.bar(labels, freight_costs, bar_width, label='运输费用', color='#2ecc71')
    ax.bar(labels, time_costs, bar_width, bottom=freight_costs, label='时间成本', color='#3498db')
    ax.bar(labels, carbon_costs, bar_width, bottom=[f + t for f, t in zip(freight_costs, time_costs)],
           label='碳排放成本', color='#e74c3c')

    ax.set_ylabel('成本 (元)')
    ax.set_title('成本构成明细 (堆叠柱状图)')
    ax.legend()

    # 在每个条形段上添加文本标签
    total = costs.total
    if total > 0:
        y_offset = costs.freight / 2
        ax.text(0, y_offset, f'{costs.freight:.2f}', ha='center', va='center', color='white', weight='bold')

        y_offset += (costs.freight / 2) + (costs.time / 2)
        ax.text(0, y_offset, f'{costs.time:.2f}', ha='center', va='center', color='white', weight='bold')
        
        y_offset += (costs.time / 2) + (costs.carbon / 2)
        ax.text(0, y_offset, f'{costs.carbon:.2f}', ha='center', va='center', color='white', weight='bold')

    plt.tight_layout()
    return fig

def plot_results_old(log_file="results/fitness_log.csv", solution_file="results/final_solution.json", test_data=None):
    """
    读取日志和解文件，生成一套完整的可视化图表。
    """
    # --- 1. 数据加载与检查 ---
    if not os.path.exists(log_file):
        print(f"错误: 日志文件 '{log_file}' 未找到。")
        return
    
    log_df = pd.read_csv(log_file)
    if log_df.empty:
        print("日志文件为空，无法生成图表。")
        return

    print(f"成功从 '{log_file}' 加载了 {len(log_df)} 条日志记录。")

    # --- 2. 准备绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 图表1: 总成本收敛曲线
    plt.figure(figsize=(12, 6))
    plt.plot(log_df['generation'], log_df['cost_total'], marker='.', linestyle='-', label='Total Cost')
    plt.title('Total Cost Convergence Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.tight_layout()

    # 图表2: 各类成本构成堆叠图
    plt.figure(figsize=(12, 6))
    plt.stackplot(log_df['generation'], log_df['cost_freight'], log_df['cost_time'], log_df['cost_carbon'],
                  labels=['Freight Cost', 'Time Cost', 'Carbon Cost'], alpha=0.8)
    plt.title('Cost Composition Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Cost Breakdown')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # 图表3: 最优路径图 (需要坐标数据)
    if test_data and 'locations_coords' in test_data and 'nodes' in test_data:
        import json
        with open(solution_file, 'r') as f:
            solution_data = json.load(f)
        path = solution_data['optimal_path']
        
        node_coords = {i: coord for i, coord in enumerate(test_data['locations_coords'])}
        node_names_map = {i: label for i, label in enumerate(test_data['nodes'])}
        
        G = nx.DiGraph()
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        G.add_nodes_from(node_coords.keys())
        G.add_edges_from(path_edges)
        
        plt.figure(figsize=(8, 8))
        nx.draw(G, node_coords, labels=node_names_map, with_labels=True, 
                node_color='lightblue', node_size=700, font_size=12, arrows=True,
                arrowstyle='->', arrowsize=20)
        plt.title('Optimal Path Found')
    else:
        print("警告: 未提供节点坐标数据，无法绘制路径图。")

    # --- 3. 显示所有图表 ---
    print("正在显示所有结果图表...")
    plt.show() 