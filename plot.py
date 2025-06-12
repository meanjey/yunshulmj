import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

def plot_results(log_file="results/fitness_log.csv", solution_file="results/final_solution.json", test_data=None):
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