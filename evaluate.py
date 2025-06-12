def evaluate_solution(individual, config, distance_matrix, transport_cost_matrix, carbon_factor_matrix):
    """
    评估单个解决方案（个体）的总成本。
    这是一个纯函数，所有依赖都通过参数传入。

    参数:
        individual (list): 代表路径的个体，例如 [0, 1, 2, 3, 4, 0]。
        config (object): 包含速度、价格等参数的配置对象。
        distance_matrix (np.array): 距离矩阵。
        transport_cost_matrix (np.array): 运输费用矩阵。
        carbon_factor_matrix (np.array): 碳排放因子矩阵。

    返回:
        tuple: (总成本, 运输费用, 时间成本, 碳排放成本)
    """
    total_transport_cost = 0
    total_distance = 0
    total_carbon_emissions = 0

    for i in range(len(individual) - 1):
        start_node = individual[i]
        end_node = individual[i+1]

        segment_distance = distance_matrix[start_node, end_node]

        # 1. 运输费用 (从矩阵读取)
        total_transport_cost += transport_cost_matrix[start_node, end_node]

        # 2. 距离累加
        total_distance += segment_distance

        # 3. 碳排放累加 (从矩阵读取因子)
        total_carbon_emissions += segment_distance * carbon_factor_matrix[start_node, end_node]

    # 计算衍生的时间成本和碳成本
    total_time_cost = (total_distance / config.speed) * config.time_cost_per_unit
    total_carbon_cost = total_carbon_emissions * config.carbon_price

    total_cost = total_transport_cost + total_time_cost + total_carbon_cost
    
    return total_cost, total_transport_cost, total_time_cost, total_carbon_cost 