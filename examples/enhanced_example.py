import numpy as np
import matplotlib.pyplot as plt
from EvoloPy.optimizers.GA import GeneticAlgorithm
from EvoloPy.benchmarks import F1, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14
from EvoloPy.benchmarks import F16, F17, F18, F20, F21, F22, F23
from EvoloPy.benchmarks import Ackley, Rosenbrock, Rastrigin, Griewank
from EvoloPy.benchmarks import CustomObjFunction

def plot_convergence(curves, labels, title):
    """Plot convergence curves"""
    plt.figure(figsize=(10, 6))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Define benchmark functions
    benchmarks = [
        F1(), F3(), F4()
    ]

    # Define custom objective function
    def custom_function(x):
        return np.sum(x**2) + np.sum(np.sin(x))

    custom_benchmark = CustomObjFunction(
        func=custom_function,
        lb=-10,
        ub=10,
        dim=30
    )
    benchmarks.append(custom_benchmark)

    # Run optimization for each benchmark
    results = []
    for benchmark in benchmarks:
        print(f"\nOptimizing {benchmark.__class__.__name__}...")

        # Create and run optimizer
        optimizer = GeneticAlgorithm(
            objective_function=benchmark,
            population_size=50,
            max_iterations=1000,
            crossover_probability=0.8,
            mutation_probability=0.1,
            elitism_count=2
        )

        result = optimizer.optimize()
        results.append(result)

        print(f"Best score: {result['best_score']}")
        print(f"Best individual: {result['best_individual']}")
        print(f"Execution time: {result['execution_time']} seconds")
    # Plot convergence curves for selected benchmarks
    selected_indices = [0, 4, 8, 12, 16, 20]  # F1, F5, F9, F13, F17, Ackley
    selected_curves = [results[i]['convergence_curve'] for i in selected_indices]
    selected_labels = [benchmarks[i].__class__.__name__ for i in selected_indices]

    plot_convergence(
        selected_curves,
        selected_labels,
        "Convergence Curves for Selected Benchmark Functions"
    )

if __name__ == "__main__":
    main()