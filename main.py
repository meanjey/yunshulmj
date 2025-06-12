from ga_solver import Config, GeneticAlgorithmVRP
from test_data import get_test_data
from plot import plot_results

def main():
    """
    项目主函数，负责协调整个流程：
    1. 加载数据
    2. 配置参数
    3. 运行求解器
    4. 可视化结果
    """
    # --- 1. 加载测试数据 ---
    print("--- 步骤 1: 加载数据 ---")
    test_data = get_test_data()
    params = test_data['parameters']
    print("数据加载完毕。")

    # --- 2. 设置配置 (算法调参入口) ---
    print("--- 步骤 2: 配置参数 ---")
    config = Config()
    
    # 从数据文件更新物理和经济参数
    config.num_locations = params['num_locations']
    config.speed = params['speed_kmh']
    config.time_cost_per_unit = params['time_cost_per_hour']
    config.carbon_price = params['carbon_price_per_kg']
    
    # GA 核心参数调优
    config.num_generations = 100
    config.population_size = 50
    config.crossover_rate = 0.85
    config.mutation_rate = 0.1
    print("参数配置完毕。")

    # --- 3. 创建GA实例并运行 ---
    print("\n--- 步骤 3: 运行优化求解器 ---")
    solver = GeneticAlgorithmVRP(
        config=config,
        distance_matrix=test_data['distance_matrix_km'],
        transport_cost_matrix=test_data['transport_cost_matrix_yuan'],
        carbon_factor_matrix=test_data['carbon_emission_factor_matrix_kg_per_km']
    )
    final_solution, solution_data = solver.run()
    print("优化流程结束。")

    # --- 4. 可视化结果 ---
    print("\n--- 步骤 4: 生成可视化报告 ---")
    plot_results(test_data=test_data)

if __name__ == '__main__':
    main()
