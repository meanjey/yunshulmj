import pandas as pd
import numpy as np

def get_test_data():
    """
    提供一组用于大件运输路径优化的简化测试数据。
    包含5个节点 (0-4)，其中节点0为仓库。
    """
    
    # 节点标签
    nodes = ['Depot', 'Node1', 'Node2', 'Node3', 'Node4']
    
    # 1. 距离矩阵 (km)
    distance_matrix = np.array([
        [0, 10, 25, 30, 15],
        [10, 0, 12, 22, 20],
        [25, 12, 0, 8, 18],
        [30, 22, 8, 0, 14],
        [15, 20, 18, 14, 0]
    ], dtype=float)

    # 2. 运输费用矩阵 (元)
    # 假设费用与距离相关，但不完全线性
    transport_cost_matrix = np.array([
        [0, 25, 65, 75, 40],
        [25, 0, 30, 55, 50],
        [65, 30, 0, 20, 45],
        [75, 55, 20, 0, 35],
        [40, 50, 45, 35, 0]
    ], dtype=float)

    # 3. 碳排放因子矩阵 (kg/km)
    # 不同路段的排放因子不同
    carbon_emission_factor_matrix = np.array([
        [0, 0.15, 0.14, 0.17, 0.16],
        [0.15, 0, 0.18, 0.15, 0.20],
        [0.14, 0.18, 0, 0.14, 0.16],
        [0.17, 0.15, 0.14, 0, 0.19],
        [0.16, 0.20, 0.16, 0.19, 0]
    ], dtype=float)

    # 新增: 用于可视化的节点坐标
    locations_coords = np.array([
        [5, 5],    # Depot
        [0, 10],   # Node1
        [15, 15],  # Node2
        [20, 10],  # Node3
        [10, 0]    # Node4
    ])

    # 4. 其他经济和物理参数
    parameters = {
        'num_locations': 5,
        'depot_index': 0,
        'speed_kmh': 60.0,            # 默认速度 (km/h)
        'time_cost_per_hour': 50.0,   # 时间成本单价 (元/小时), 沿用之前设定
        'carbon_price_per_kg': 50.0   # 碳价 (元/kg)
    }

    # 组织成一个字典返回
    test_data = {
        "nodes": nodes,
        "locations_coords": locations_coords,
        "distance_matrix_km": distance_matrix,
        "transport_cost_matrix_yuan": transport_cost_matrix,
        "carbon_emission_factor_matrix_kg_per_km": carbon_emission_factor_matrix,
        "parameters": parameters
    }

    return test_data

def get_test_data_as_dataframes(data_dict=None):
    """将测试数据字典转换为Pandas DataFrame格式，便于查看。"""
    if data_dict is None:
        data_dict = get_test_data()
        
    nodes = data_dict['nodes']
    
    df_distance = pd.DataFrame(data_dict['distance_matrix_km'], index=nodes, columns=nodes)
    df_cost = pd.DataFrame(data_dict['transport_cost_matrix_yuan'], index=nodes, columns=nodes)
    df_carbon_factor = pd.DataFrame(data_dict['carbon_emission_factor_matrix_kg_per_km'], index=nodes, columns=nodes)
    
    return {
        "distance": df_distance,
        "transport_cost": df_cost,
        "carbon_factor": df_carbon_factor,
        "parameters": pd.Series(data_dict['parameters'])
    }

if __name__ == '__main__':
    # --- 方案一: 获取原生字典和Numpy数组 ---
    print("--- 1. Python 字典与 Numpy 数组格式 ---")
    data = get_test_data()
    print("参数:")
    print(data['parameters'])
    print("\n距离矩阵 (km):")
    print(data['distance_matrix_km'])
    print("\n运输费用矩阵 (元):")
    print(data['transport_cost_matrix_yuan'])
    print("\n碳排放因子矩阵 (kg/km):")
    print(data['carbon_emission_factor_matrix_kg_per_km'])
    print("-" * 40)

    # --- 方案二: 获取Pandas DataFrame格式 (更易读) ---
    print("\n--- 2. Pandas DataFrame 格式 ---")
    data_dfs = get_test_data_as_dataframes(data)
    for name, df in data_dfs.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(df) 