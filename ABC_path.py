import networkx as nx
import numpy as np
import random

class DiscreteABCShortestPath:
    def __init__(self, G, source, target, num_bees=20, max_iterations=100):
        # Khoi tao thuat toan ABC roi rac cho bai toan duong di
        # G: Do thi NetworkX
        # source, target: Nut bat dau va ket thuc
        self.G = G
        self.source = source
        self.target = target
        self.num_bees = num_bees
        self.max_iterations = max_iterations
        
        # Phan chia dan ong: 50 percent Tho, 50 percent Quan sat
        self.num_employed = num_bees // 2
        self.num_onlooker = num_bees - self.num_employed
        
        # Khoi tao quan the
        self.population = []
        self.fitness_values = []
        self.initialize_population()
        
        # Luu giai phap tot nhat (Global Best)
        self.best_solution = None
        self.best_cost = float('inf')
        self.update_global_best()

    def calculate_path_cost(self, path):
        # Tinh tong trong so (Objective Function - Minimize)
        cost = 0
        try:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if self.G.has_edge(u, v):
                    cost += self.G[u][v].get('weight', 1)
                else:
                    return float('inf') # Duong di khong hop le
            return cost
        except:
            return float('inf')

    def cost_to_fitness(self, cost):
        # Chuyen doi Cost sang Fitness (Cang lon cang tot)
        if cost == float('inf'): return 0.0
        return 1.0 / (1.0 + cost)

    def random_walk_path(self):
        # Tao path random tu source den target
        path = [self.source]
        curr = self.source
        visited = {self.source}
        
        while curr != self.target:
            neighbors = [n for n in self.G.neighbors(curr) if n not in visited]
            if not neighbors: return None # Gap ngo cut
            curr = random.choice(neighbors)
            path.append(curr)
            visited.add(curr)
        return path

    def initialize_population(self):
        # Khoi tao quan the ban dau
        count = 0
        attempts = 0
        while count < self.num_bees and attempts < self.num_bees * 100:
            path = self.random_walk_path()
            if path:
                self.population.append(path)
                self.fitness_values.append(self.cost_to_fitness(self.calculate_path_cost(path)))
                count += 1
            attempts += 1
        
        # Lap day neu thieu (bang cach sao chep)
        if 0 < count < self.num_bees:
            while len(self.population) < self.num_bees:
                idx = random.randint(0, count - 1)
                self.population.append(self.population[idx][:])
                self.fitness_values.append(self.fitness_values[idx])

    def mutate_path(self, path):
        # Cau truc lang gieng: Cat mot doan va noi lai bang duong khac
        if len(path) < 3: return path
        try:
            idx1, idx2 = sorted(random.sample(range(len(path)), 2))
            if idx2 - idx1 < 2: return path
            
            sub_start, sub_end = path[idx1], path[idx2]
            
            # Tim duong di thay the cuc bo (Local Search)
            try:
                alternatives = list(nx.all_simple_paths(self.G, sub_start, sub_end, cutoff=len(path)))
                if len(alternatives) > 1:
                    new_segment = random.choice(alternatives)
                    new_path = path[:idx1] + new_segment + path[idx2+1:]
                    return new_path
            except: pass
        except: pass
        return path

    def update_global_best(self):
        best_idx = np.argmax(self.fitness_values)
        current_cost = self.calculate_path_cost(self.population[best_idx])
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_solution = self.population[best_idx][:]

    def run(self):
        limit = [0] * self.num_bees
        max_limit = 5 # Nguong cho Ong Trinh Sat (Scout Bee)
        
        for iteration in range(self.max_iterations):
            # --- GIAI DOAN ONG THO (EMPLOYED BEES) ---
            for i in range(self.num_employed):
                candidate = self.mutate_path(self.population[i])
                candidate_fit = self.cost_to_fitness(self.calculate_path_cost(candidate))
                
                if candidate_fit > self.fitness_values[i]:
                    self.population[i] = candidate
                    self.fitness_values[i] = candidate_fit
                    limit[i] = 0
                else:
                    limit[i] += 1
            
            # --- GIAI DOAN ONG QUAN SAT (ONLOOKER BEES) ---
            total_fit = sum(self.fitness_values)
            if total_fit > 0:
                probs = [f/total_fit for f in self.fitness_values]
            else:
                probs = [1.0/self.num_bees]*self.num_bees
            
            for i in range(self.num_onlooker):
                selected_idx = np.random.choice(range(self.num_bees), p=probs)
                candidate = self.mutate_path(self.population[selected_idx])
                candidate_fit = self.cost_to_fitness(self.calculate_path_cost(candidate))
                
                if candidate_fit > self.fitness_values[selected_idx]:
                    self.population[selected_idx] = candidate
                    self.fitness_values[selected_idx] = candidate_fit
                    limit[selected_idx] = 0
                else:
                    limit[selected_idx] += 1
            
            # --- GIAI DOAN ONG TRINH SAT (SCOUT BEES) ---
            for i in range(self.num_bees):
                if limit[i] > max_limit:
                    new_path = self.random_walk_path()
                    if new_path:
                        self.population[i] = new_path
                        self.fitness_values[i] = self.cost_to_fitness(self.calculate_path_cost(new_path))
                        limit[i] = 0
            
            self.update_global_best()
            
        return self.best_solution, self.best_cost