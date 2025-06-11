"""
Created on Sat May 24 20:18:05 2024

@author: Raneem
"""
import numpy as np
import random
import time
import sys

from EvoloPy.solution import solution
from ..benchmarks import BaseBenchmark


class GeneticAlgorithm:
    """Genetic Algorithm optimizer"""

    def __init__(self, objective_function, population_size=50, max_iterations=1000,
                 crossover_probability=0.8, mutation_probability=0.1, elitism_count=2):
        """
        Initialize the Genetic Algorithm optimizer

        Parameters:
        -----------
        objective_function : BaseBenchmark
            The objective function to optimize
        population_size : int, optional
            Size of the population, default is 50
        max_iterations : int, optional
            Maximum number of iterations, default is 1000
        crossover_probability : float, optional
            Probability of crossover, default is 0.8
        mutation_probability : float, optional
            Probability of mutation, default is 0.1
        elitism_count : int, optional
            Number of best individuals to keep, default is 2
        """
        if not isinstance(objective_function, BaseBenchmark):
            raise TypeError("objective_function must be an instance of BaseBenchmark")

        self.objective_function = objective_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_count = elitism_count

        # Initialize population
        if not isinstance(self.objective_function.lb, list):
            self.lb = [self.objective_function.lb] * self.objective_function.dim
        if not isinstance(self.objective_function.ub, list):
            self.ub = [self.objective_function.ub] * self.objective_function.dim

        # Initialize population
        self.population = np.zeros((self.population_size, self.objective_function.dim))

        # Initialize scores
        self.scores = np.random.uniform(0.0, 1.0, self.population_size)

        # Initialize best individual
        self.best_individual = np.zeros(self.objective_function.dim)
        self.best_score = float("inf")

        # Initialize convergence curve
        self.convergence_curve = np.zeros(self.max_iterations)

        for i in range(self.objective_function.dim):
            # For each dimension `i`, assign random values to the population for that dimension.
            # The random values are scaled to the range defined by the lower bound `lb[i]` and upper bound `ub[i]`.
            self.population[:, i] = np.random.uniform(0, 1, self.population_size) * (self.ub[i] - self.lb[i]) + self.lb[i]

    def _calculate_scores(self):
        """Calculate scores for all individuals in the population"""
        scores = np.full(self.population_size, np.inf)

    # Step 2: Loop through each individual in the population
        for i in range(0, self.population_size):
            # Step 3: Ensure that each individual is within the defined bounds of the search space
            self.population[i] = np.clip(self.population[i], self.lb, self.ub)

            # Step 4: Calculate the fitness value (objective function) for each individual
            scores[i] = self.objective_function.evaluate(self.population[i, :])

    def _sort_population(self):
        """Sort population based on scores"""
        sorted_idx = np.argsort(self.scores)
        self.population = self.population[sorted_idx]
        self.scores = self.scores[sorted_idx]

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if np.random.random() < self.crossover_probability:
            # Single point crossover
            point = np.random.randint(0, self.objective_function.dim)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()


    def _pairSelection(self, population, scores, popSize):
        """
        This is used to select one pair of parents using roulette Wheel Selection mechanism

        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population

        Returns
        -------
        list
            parent1: The first parent individual of the pair
        list
            parent2: The second parent individual of the pair
        """

        def rouletteWheelSelectionId(inverted_scores, popSize):

            # Step 1: Check if all scores are identical
            if len(set(inverted_scores)) == 1:
                # If all scores are the same, perform random selection
                return random.randint(0, popSize - 1)

            # Step 2: Calculate the total fitness of the population
            total_fitness = sum(inverted_scores)

            # Step 3: Normalize the inverted scores to create probabilities
            normalized_scores = [score / total_fitness for score in inverted_scores]

            # Step 4: Generate cumulative probabilities for roulette wheel
            cumulative_probs = np.cumsum(normalized_scores)

            # Step 5: Select a random number between 0 and 1
            random_num = random.random()

            # Step 6: Find the first index where the random number is less than or equal to the cumulative probability
            for i, cumulative_prob in enumerate(cumulative_probs):
                if random_num <= cumulative_prob:
                    return i

        # Step A: Invert scores so lower scores (better fitness) have higher probabilities
        max_score = max(scores)
        inverted_scores = [max_score - score for score in scores]

        # Step B: Select the first parent using the roulette wheel mechanism
        parent1Id = rouletteWheelSelectionId(inverted_scores, popSize)
        parent1 = population[parent1Id].copy()  # Copy the selected parent to avoid altering the original population

        # Step C: Select the second parent, ensuring it's different from the first parent
        parent2Id = parent1Id
        while parent2Id == parent1Id:  # Keep selecting until a different parent is chosen
            parent2Id = rouletteWheelSelectionId(inverted_scores, popSize)
        parent2 = population[parent2Id].copy()  # Copy the selected parent to avoid altering the original population

        # Return the selected pair of parents
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual):
        """Mutate an individual"""
        mutationIndex = random.randint(0, len(individual) - 1)

        individual[mutationIndex] = random.uniform(self.lb[mutationIndex], self.ub[mutationIndex])
        return individual

    def _clear_duplicates(self):
        """Clear duplicate individuals from the population"""
        newPopulation = np.unique(self.population, axis=0)
        oldLen = len(self.population)
        newLen = len(newPopulation)

        if newLen < oldLen:
        # Calculate how many duplicates were removed
            nDuplicates = oldLen - newLen

            # Step 4: Generate random new individuals to replace the duplicates
            randomIndividuals = np.random.uniform(0, 1, (nDuplicates, len(self.population[0]))) * (np.array(self.ub) - np.array(self.lb)) + np.array(self.lb)

            # Step 5: Append the random individuals to the new population
            newPopulation = np.append(newPopulation, randomIndividuals, axis=0)
            self.population = newPopulation

    def calculateCost(self):

        """
        It calculates the fitness value of each individual in the population

        Parameters
        ----------
        objf : function
            The objective function selected
        population : list
            The list of individuals
        popSize: int
            Number of chrmosomes in a population
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list

        Returns
        -------
        list
            scores: fitness values of all individuals in the population
        """

        # Step 1: Initialize an array to store fitness values, with each set to infinity initially
        scores = np.full(self.population_size, np.inf)

        # Step 2: Loop through each individual in the population
        for i in range(0, self.population_size):
            # Step 3: Ensure that each individual is within the defined bounds of the search space
            self.population[i] = np.clip(self.population[i], self.lb, self.ub)

            # Step 4: Calculate the fitness value (objective function) for each individual
            self.scores[i] = self.objective_function.evaluate(self.population[i, :])

    def optimize(self):
        """Run the genetic algorithm optimization"""

        print('GA is optimizing  "' + self.objective_function.__class__.__name__ + '"')

        self.timerStart = time.time()
        self.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for iteration in range(self.max_iterations):
            self.population = self._crossover_population()
            self._mutate_population()
            self._clear_duplicates()

            self.calculateCost()

            self._sort_population()

            # Update best individual
            self.best_score = min(self.scores)
            self.best_individual = self.population[0].copy()
            self.convergence_curve[iteration] = self.best_score
        self.endTime = time.time()
        self.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.executionTime = self.endTime - self.startTime


        return {
            'best_individual': self.best_individual,
            'best_score': self.best_score,
            'convergence_curve': self.convergence_curve,
            'execution_time': self.executionTime
        }

    def _crossover_population(self):
        """Perform crossover on the entire population"""
        new_population = np.zeros_like(self.population)
        new_population[0:self.elitism_count] = self.population[0:self.elitism_count]

        for i in range(self.elitism_count, self.population_size, 2):
            # Select parents
            parent1, parent2 = self._pairSelection(self.population, self.scores, self.population_size)
            individualLength = len(parent1)

            # Perform crossover
            if random.random() < self.crossover_probability:
                offspring1, offspring2 = self._crossover(parent1, parent2)  # Perform crossover to produce two offspring
            else:
                # If no crossover occurs, the offspring are copies of the parents
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            # Add to new population
            new_population[i] = offspring1
            new_population[i + 1] = offspring2

        return new_population

    def _mutate_population(self):
        """Mutate the entire population"""
        for i in range(self.elitism_count, self.population_size):
            if np.random.random() < self.mutation_probability:
                self._mutate(self.population[i])