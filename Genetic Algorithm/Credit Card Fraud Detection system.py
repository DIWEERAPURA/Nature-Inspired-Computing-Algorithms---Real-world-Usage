import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def load_data(file_path):
    """
    Load dataset from a CSV file.
    Expected structure: feature columns and a 'Class' column (1 for fraud, 0 for non-fraud).
    """
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def preprocess_data(data):
    """
    Separate features and target from the dataset.
    Optionally include scaling or other preprocessing steps.
    """
    # Assumes the target column is named 'Class'
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Uncomment and modify the following if scaling is required:
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y


def initialize_population(pop_size, n_features):
    """
    Create an initial population of random binary chromosomes.
    Each chromosome is an array where 1 indicates the feature is selected.
    """
    return [np.random.randint(0, 2, n_features) for _ in range(pop_size)]


def evaluate_fitness(chromosome, X, y, cv_splits=5):
    """
    Evaluate an individual chromosome by calculating the cross-validated F1 score.
    A chromosome with no selected features receives a fitness of 0.
    """
    if np.count_nonzero(chromosome) == 0:
        return 0
    # Select features where the chromosome has a '1'
    selected_features = X.columns[np.where(chromosome == 1)[0]]
    X_selected = X[selected_features]

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    # Use F1 score to account for class imbalance in fraud detection
    scores = cross_val_score(model, X_selected, y, cv=skf, scoring='f1')
    return scores.mean()


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Perform tournament selection: randomly choose individuals and return the one with the highest fitness.
    """
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = selected_indices[0]
    best_fitness = fitnesses[best_idx]
    for idx in selected_indices:
        if fitnesses[idx] > best_fitness:
            best_idx = idx
            best_fitness = fitnesses[idx]
    return population[best_idx]


def crossover(parent1, parent2):
    """
    Perform uniform crossover between two parents to produce an offspring.
    """
    child = np.empty_like(parent1)
    for i in range(len(parent1)):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child


def mutation(chromosome, mutation_rate=0.01):
    """
    Mutate a chromosome by flipping bits with a given mutation rate.
    """
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


def genetic_algorithm(X, y, pop_size=50, generations=30, mutation_rate=0.02, elitism=2):
    """
    Run the Genetic Algorithm for feature selection.
    """
    n_features = X.shape[1]
    population = initialize_population(pop_size, n_features)

    best_chromosome = None
    best_fitness = -np.inf

    for gen in range(generations):
        # Evaluate current population
        fitnesses = [evaluate_fitness(chrom, X, y) for chrom in population]

        # Track the best solution
        for i, fit in enumerate(fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_chromosome = population[i].copy()

        print(f"Generation {gen + 1}: Best F1 Score = {best_fitness:.4f}")

        # Elitism: Retain top 'elitism' individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population = [population[idx].copy() for idx in sorted_indices[:elitism]]

        # Generate new individuals until population is refilled
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_chromosome, best_fitness


def main():
    """
    Main function to run the Genetic Algorithm for Credit Card Fraud Detection.
    Replace 'path_to_dataset.csv' with the actual path to your dataset.
    """
    file_path = "path_to_dataset.csv"  # <-- Update with your dataset file path
    data = load_data(file_path)
    if data is None:
        print("Dataset not loaded. Exiting.")
        return

    X, y = preprocess_data(data)
    print(f"Dataset contains {X.shape[0]} samples and {X.shape[1]} features.")

    best_chromosome, best_fitness = genetic_algorithm(X, y, pop_size=50, generations=30, mutation_rate=0.02, elitism=2)
    selected_features = X.columns[np.where(best_chromosome == 1)[0]]

    print("\nOptimal feature subset selected by the Genetic Algorithm:")
    print(selected_features.tolist())
    print(f"Achieved cross-validated F1 Score: {best_fitness:.4f}")


if __name__ == "__main__":
    main()
