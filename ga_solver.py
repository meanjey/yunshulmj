import numpy as np
import random
import os
import csv
import json
from evaluate import evaluate_solution

class Config:
    """算法配置类"""
    def __init__(self):
        # 问题定义
        self.num_locations = 20
        self.depot_index = 0

        # GA 核心参数调优
        self.num_generations = 100
        self.population_size = 50
        self.crossover_rate = 0.85
        self.mutation_rate = 0.1
        self.tournament_size = 5
        self.elitism_size = 2

        # 物理和经济参数
        self.speed = 60.0
        self.time_cost_per_unit = 50.0
        self.carbon_emission_factor = 0.16
        self.carbon_price = 0.5

        # 模拟退火 (SA) 参数
        self.sa_initial_temp = 100.0
        self.sa_final_temp = 0.1
        self.sa_cooling_rate = 0.995

class GeneticAlgorithmVRP:
    """遗传算法与模拟退火混合求解器"""
    def __init__(self, config, distance_matrix, transport_cost_matrix, carbon_factor_matrix):
        self.config = config
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.carbon_factor_matrix = carbon_factor_matrix
        
        self.population = []
        self.best_individual = None
        self.best_fitness = -1
        self.log_filepath = "results/fitness_log.csv"

    def calculate_fitness(self, individual):
        total_cost, _, _, _ = evaluate_solution(
            individual, self.config, self.distance_matrix, 
            self.transport_cost_matrix, self.carbon_factor_matrix
        )
        # 避免除以零和负数
        if total_cost is None or total_cost <= 0 or total_cost == float('inf'):
            return 1e-6
        return 1 / total_cost

    def create_individual(self):
        locations = list(range(self.config.num_locations))
        locations.remove(self.config.depot_index)
        random.shuffle(locations)
        return [self.config.depot_index] + locations + [self.config.depot_index]

    def initialize_population(self):
        self.population = [self.create_individual() for _ in range(self.config.population_size)]

    def selection(self, population_with_fitness):
        selected = []
        for _ in range(len(self.population)):
            tournament = random.sample(population_with_fitness, self.config.tournament_size)
            winner = max(tournament, key=lambda item: item[1])
            selected.append(winner[0])
        return selected

    def crossover(self, parent1, parent2):
        if random.random() > self.config.crossover_rate:
            return list(parent1), list(parent2)

        size = len(parent1) - 2
        
        # 创建子代1
        child1_genes = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child1_genes[start:end+1] = parent1[start+1:end+2]
        p2_genes = [gene for gene in parent2[1:-1] if gene not in child1_genes]
        idx = 0
        for i in range(size):
            if child1_genes[i] == -1:
                child1_genes[i] = p2_genes[idx]
                idx += 1
        child1 = [self.config.depot_index] + child1_genes + [self.config.depot_index]

        # 创建子代2
        child2_genes = [-1] * size
        child2_genes[start:end+1] = parent2[start+1:end+2]
        p1_genes = [gene for gene in parent1[1:-1] if gene not in child2_genes]
        idx = 0
        for i in range(size):
            if child2_genes[i] == -1:
                child2_genes[i] = p1_genes[idx]
                idx += 1
        child2 = [self.config.depot_index] + child2_genes + [self.config.depot_index]
        
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.config.mutation_rate:
            size = len(individual) - 2
            if size > 1:
                idx1, idx2 = random.sample(range(1, size + 1), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def get_neighbor(self, solution):
        neighbor = solution[:]
        size = len(neighbor) - 2
        if size > 1:
            idx1, idx2 = random.sample(range(1, size + 1), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def simulated_annealing(self, initial_solution):
        current_solution = initial_solution
        best_solution = current_solution
        current_cost, _, _, _ = evaluate_solution(
            current_solution, self.config, self.distance_matrix,
            self.transport_cost_matrix, self.carbon_factor_matrix
        )
        best_cost = current_cost
        T = self.config.sa_initial_temp

        while T > self.config.sa_final_temp:
            neighbor = self.get_neighbor(current_solution)
            neighbor_cost, _, _, _ = evaluate_solution(
                neighbor, self.config, self.distance_matrix,
                self.transport_cost_matrix, self.carbon_factor_matrix
            )
            delta_e = neighbor_cost - current_cost
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_solution = neighbor
                current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
            T *= self.config.sa_cooling_rate
        return best_solution

    def run(self):
        if not os.path.exists('results'):
            os.makedirs('results')
        with open(self.log_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'cost_total', 'cost_freight', 'cost_time', 'cost_carbon'])
        
        self.initialize_population()
        print("遗传算法开始运行...")

        for generation in range(self.config.num_generations):
            population_with_fitness = [(ind, self.calculate_fitness(ind)) for ind in self.population]
            all_costs = [1 / fit for ind, fit in population_with_fitness]
            avg_cost = np.mean(all_costs)
            
            current_best_individual, current_best_fitness = max(population_with_fitness, key=lambda item: item[1])
            costs_breakdown = evaluate_solution(
                current_best_individual, self.config, self.distance_matrix,
                self.transport_cost_matrix, self.carbon_factor_matrix
            )
            current_best_cost = costs_breakdown[0]

            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = list(current_best_individual)
            
            print(f"代 {generation + 1: >3} | 最优成本: {current_best_cost:.2f} | 平均成本: {avg_cost:.2f}")
            with open(self.log_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([generation + 1] + list(costs_breakdown))
            
            next_generation = [item[0] for item in sorted(population_with_fitness, key=lambda x: x[1], reverse=True)[:self.config.elitism_size]]
            selected_parents = self.selection(population_with_fitness)
            while len(next_generation) < self.config.population_size:
                p1, p2 = random.sample(selected_parents, 2)
                c1, c2 = self.crossover(p1, p2)
                next_generation.append(self.mutate(c1))
                if len(next_generation) < self.config.population_size:
                    next_generation.append(self.mutate(c2))
            self.population = next_generation

        print("\n遗传算法运行结束。")
        ga_costs = evaluate_solution(self.best_individual, self.config, self.distance_matrix, self.transport_cost_matrix, self.carbon_factor_matrix)
        
        print("\n开始模拟退火优化...")
        final_solution = self.simulated_annealing(list(self.best_individual))
        final_costs = evaluate_solution(final_solution, self.config, self.distance_matrix, self.transport_cost_matrix, self.carbon_factor_matrix)
        print(f"模拟退火完成。")

        solution_data = {"optimal_path": final_solution, "costs": {
            "total": round(final_costs[0], 4), "freight": round(final_costs[1], 4),
            "time": round(final_costs[2], 4), "carbon": round(final_costs[3], 4)}}
        solution_filepath = "results/final_solution.json"
        with open(solution_filepath, 'w') as f:
            json.dump(solution_data, f, indent=4)
        
        print(f"\n最终优化方案已保存至: {solution_filepath}")
        return final_solution, solution_data 

def solve_vrp(df, carbon_price=50, time_cost=60):
    """求解VRP问题的主函数
    
    Args:
        df: pandas DataFrame，包含from, to, cost, distance, carbon_factor列
        carbon_price: float，碳价（元/kg）
        time_cost: float，时间成本（元/h）
    
    Returns:
        dict: 包含最优路径和成本信息的字典
    """
    # 创建配置对象
    config = Config()
    config.carbon_price = carbon_price
    config.time_cost_per_unit = time_cost
    
    # 获取唯一的地点列表
    locations = sorted(list(set(df['from'].unique()) | set(df['to'].unique())))
    location_to_index = {loc: idx for idx, loc in enumerate(locations)}
    config.num_locations = len(locations)
    
    # 创建距离矩阵、运输成本矩阵和碳排放因子矩阵
    distance_matrix = np.zeros((config.num_locations, config.num_locations))
    transport_cost_matrix = np.zeros((config.num_locations, config.num_locations))
    carbon_factor_matrix = np.zeros((config.num_locations, config.num_locations))
    
    for _, row in df.iterrows():
        i = location_to_index[row['from']]
        j = location_to_index[row['to']]
        distance_matrix[i, j] = row['distance']
        transport_cost_matrix[i, j] = row['cost']
        carbon_factor_matrix[i, j] = row['carbon_factor']
    
    # 创建并运行遗传算法求解器
    ga = GeneticAlgorithmVRP(config, distance_matrix, transport_cost_matrix, carbon_factor_matrix)
    final_solution, solution_data = ga.run()
    
    # 将数字索引转换回地点名称
    path = [locations[i] for i in final_solution]
    
    # 创建结果对象
    class Result:
        def __init__(self, path, costs):
            self.path = path
            self.costs = costs
    
    costs = type('Costs', (), {
        'freight': solution_data['costs']['freight'],
        'time': solution_data['costs']['time'],
        'carbon': solution_data['costs']['carbon'],
        'total': solution_data['costs']['total']
    })
    
    return Result(path, costs) 