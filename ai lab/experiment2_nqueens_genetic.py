"""
Experiment 2: Genetic Algorithm for the N-Queens Problem
Finds a valid arrangement of N queens on an N×N chessboard with no attacks.
"""

import random
import time

class NQueensGenetic:
    def __init__(self, n, population_size=100, mutation_rate=0.1, max_generations=1000, elitism_rate=0.1):
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.elitism_rate = elitism_rate
        self.elitism_count = int(population_size * elitism_rate)

    def create_individual(self):
        """Create a random chromosome (one queen per column, row position varies)"""
        return random.sample(range(self.n), self.n)

    def create_population(self):
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]

    def calculate_fitness(self, chromosome):
        """
        Fitness function: number of non-attacking queen pairs
        Maximum fitness = n*(n-1)/2 (all pairs non-attacking)
        """
        attacks = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Check if queens attack each other
                if chromosome[i] == chromosome[j]:  # Same row
                    attacks += 1
                elif abs(chromosome[i] - chromosome[j]) == abs(i - j):  # Same diagonal
                    attacks += 1
        max_pairs = self.n * (self.n - 1) // 2
        return max_pairs - attacks

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select parent using tournament selection"""
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        crossover_point = random.randint(1, self.n - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Fix duplicates using crossover with repair
        child1 = self.repair_chromosome(child1)
        child2 = self.repair_chromosome(child2)

        return child1, child2

    def repair_chromosome(self, chromosome):
        """Repair chromosome to ensure valid permutation (no duplicates)"""
        seen = set()
        result = []
        for gene in chromosome:
            if gene not in seen:
                seen.add(gene)
                result.append(gene)
        # Add missing genes
        for gene in range(self.n):
            if gene not in seen:
                result.append(gene)
        return result

    def mutate(self, chromosome):
        """Swap mutation - swap two random positions"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def evolve(self, population):
        """Create next generation with elitism, selection, crossover, mutation"""
        # Calculate fitness for all individuals
        fitness_scores = [self.calculate_fitness(ind) for ind in population]

        # Sort by fitness (descending)
        sorted_population = [x for x, _ in sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)]

        # Elitism: keep best individuals
        new_population = sorted_population[:self.elitism_count]

        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])

        return new_population[:self.population_size]

    def solve(self):
        """Run genetic algorithm to find solution"""
        population = self.create_population()
        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(self.max_generations):
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            best_individual = population[fitness_scores.index(best_fitness)]

            # Check if solved (max fitness = n*(n-1)/2)
            max_fitness = self.n * (self.n - 1) // 2
            if best_fitness == max_fitness:
                return {
                    'solution': best_individual,
                    'generation': generation + 1,
                    'best_fitness_history': best_fitness_history,
                    'avg_fitness_history': avg_fitness_history,
                    'success': True
                }

            population = self.evolve(population)

        # Return best found if max generations reached
        fitness_scores = [self.calculate_fitness(ind) for ind in population]
        best_individual = population[fitness_scores.index(max(fitness_scores))]

        return {
            'solution': best_individual,
            'generation': self.max_generations,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'success': False
        }

    def print_board(self, chromosome):
        """Print the chessboard with queens"""
        for row in range(self.n):
            line = ""
            for col in range(self.n):
                if chromosome[col] == row:
                    line += "Q "
                else:
                    line += ". "
            print(line)
        print()

def main():
    print("=" * 70)
    print("Genetic Algorithm for N-Queens Problem")
    print("=" * 70)

    # Test with different N values
    test_configs = [
        {'n': 4, 'population_size': 50, 'mutation_rate': 0.1, 'max_generations': 500},
        {'n': 8, 'population_size': 100, 'mutation_rate': 0.1, 'max_generations': 1000},
        {'n': 12, 'population_size': 150, 'mutation_rate': 0.15, 'max_generations': 2000},
    ]

    for config in test_configs:
        n = config['n']
        print(f"\n{'=' * 70}")
        print(f"Solving {n}-Queens Problem")
        print(f"Population Size: {config['population_size']}, Mutation Rate: {config['mutation_rate']}")
        print(f"Max Generations: {config['max_generations']}")
        print("=" * 70)

        ga = NQueensGenetic(
            n=n,
            population_size=config['population_size'],
            mutation_rate=config['mutation_rate'],
            max_generations=config['max_generations']
        )

        start_time = time.time()
        result = ga.solve()
        end_time = time.time()

        print(f"\nTime taken: {end_time - start_time:.4f} seconds")
        print(f"Generations: {result['generation']}")
        print(f"Success: {'YES' if result['success'] else 'NO (limit reached)'}")

        if result['success']:
            print(f"\nSolution found in {result['generation']} generations:")
            ga.print_board(result['solution'])
        else:
            print(f"\nBest solution found (not optimal):")
            ga.print_board(result['solution'])
            fitness = ga.calculate_fitness(result['solution'])
            max_fitness = n * (n - 1) // 2
            print(f"Fitness: {fitness}/{max_fitness}")

        # Test different mutation rates for 8-queens
        if n == 8:
            print(f"\n{'=' * 70}")
            print("Parameter Sensitivity Analysis for 8-Queens")
            print("=" * 70)

            mutation_rates = [0.05, 0.1, 0.2, 0.3]
            for mr in mutation_rates:
                ga_test = NQueensGenetic(
                    n=8,
                    population_size=100,
                    mutation_rate=mr,
                    max_generations=1000
                )
                result_test = ga_test.solve()
                print(f"Mutation Rate {mr:4.1%}: {'Success' if result_test['success'] else 'Failed'} in {result_test['generation']} generations")

if __name__ == "__main__":
    main()
