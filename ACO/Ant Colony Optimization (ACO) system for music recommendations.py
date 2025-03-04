import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score
import random
import time


class ACORecommender:
    """
    Ant Colony Optimization for Music Recommendations.

    The recommender builds a pheromone matrix to guide a swarm of ants (simulated recommendation paths)
    over a set of songs. Visibility between songs is computed as the inverse of the Euclidean distance
    between their normalized feature vectors. Each ant builds a path (a list of recommended songs)
    by probabilistically choosing the next song based on the pheromone level and visibility.

    Parameters:
        data (pd.DataFrame): Dataset containing song features.
        feature_cols (list): List of column names representing song features used for similarity.
        popularity_col (str): Column name representing song popularity.
        n_ants (int): Number of ants (paths) to simulate per iteration.
        n_iterations (int): Number of iterations for the optimization process.
        alpha (float): Influence of the pheromone.
        beta (float): Influence of the visibility (similarity).
        rho (float): Pheromone evaporation rate.
        Q (float): Pheromone deposit factor.
    """

    def __init__(self, data, feature_cols, popularity_col,
                 n_ants=10, n_iterations=50, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.popularity_col = popularity_col
        self.n_songs = len(data)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        # Normalize features for similarity calculation
        self._preprocess_data()

        # Initialize pheromone matrix with a small constant value (0.1)
        self.pheromone = np.full((self.n_songs, self.n_songs), 0.1)

        # Compute visibility matrix (heuristic information)
        self.visibility = self._compute_visibility_matrix()

        # Keep track of best path and its quality
        self.best_path = None
        self.best_quality = -np.inf
        self.quality_progression = []

    def _preprocess_data(self):
        """Normalize feature columns using MinMaxScaler."""
        scaler = MinMaxScaler()
        self.data[self.feature_cols] = scaler.fit_transform(self.data[self.feature_cols])
        # Optionally convert popularity to numeric if needed
        self.data[self.popularity_col] = pd.to_numeric(self.data[self.popularity_col], errors='coerce')

    def _compute_visibility_matrix(self):
        """
        Compute the visibility (heuristic desirability) between songs.
        We use the inverse of the Euclidean distance between feature vectors.
        A small constant is added to the distance to avoid division by zero.
        """
        features = self.data[self.feature_cols].values
        visibility = np.zeros((self.n_songs, self.n_songs))
        for i in range(self.n_songs):
            for j in range(self.n_songs):
                if i != j:
                    distance = np.linalg.norm(features[i] - features[j]) + 1e-6
                    visibility[i][j] = 1.0 / distance
                else:
                    visibility[i][j] = 0  # No self-loop
        return visibility

    def _select_next_song(self, current_song, visited):
        """
        Probabilistically select the next song based on pheromone and visibility.
        Excludes songs that have already been visited.
        """
        probabilities = []
        for j in range(self.n_songs):
            if j in visited:
                probabilities.append(0)
            else:
                # Calculate probability based on pheromone^alpha * visibility^beta
                pheromone_value = self.pheromone[current_song][j] ** self.alpha
                visibility_value = self.visibility[current_song][j] ** self.beta
                probabilities.append(pheromone_value * visibility_value)
        probabilities = np.array(probabilities)
        # Normalize probabilities
        if probabilities.sum() == 0:
            probabilities = np.ones(self.n_songs)
            probabilities[list(visited)] = 0
        probabilities = probabilities / probabilities.sum()
        next_song = np.random.choice(range(self.n_songs), p=probabilities)
        return next_song

    def _ant_construct_path(self):
        """
        Each ant constructs a recommendation path.
        Here we assume a path length of a fixed number (e.g., 10 recommendations).
        """
        path_length = min(10, self.n_songs)  # Adjust as needed
        # Start from a random song
        current_song = random.randint(0, self.n_songs - 1)
        path = [current_song]
        while len(path) < path_length:
            next_song = self._select_next_song(current_song, visited=set(path))
            path.append(next_song)
            current_song = next_song
        return path

    def _calculate_quality(self, path):
        """
        Evaluate the quality of a recommendation path.
        For demonstration, quality is computed as the sum of popularity scores of the recommended songs.
        In a real system, you might compare recommendations against a ground-truth set using precision and recall.
        """
        quality = self.data.iloc[path][self.popularity_col].sum()
        return quality

    def _update_pheromone(self, all_paths, all_qualities):
        """
        Update pheromone matrix with evaporation and deposit from all ants.
        """
        # Evaporate pheromone on all edges
        self.pheromone = (1 - self.rho) * self.pheromone

        # Deposit pheromone for each ant's path proportional to its quality
        for path, quality in zip(all_paths, all_qualities):
            deposit_amount = self.Q * quality
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += deposit_amount
                # Optionally, make pheromone symmetric
                self.pheromone[path[i + 1]][path[i]] += deposit_amount

    def run_optimization(self):
        """
        Run the ACO optimization over a fixed number of iterations.
        At each iteration, multiple ants construct recommendation paths.
        The best path (highest quality) is tracked, and pheromone trails are updated accordingly.
        """
        start_time = time.time()
        for iteration in range(self.n_iterations):
            all_paths = []
            all_qualities = []
            for ant in range(self.n_ants):
                path = self._ant_construct_path()
                quality = self._calculate_quality(path)
                all_paths.append(path)
                all_qualities.append(quality)
                # Track best overall path
                if quality > self.best_quality:
                    self.best_quality = quality
                    self.best_path = path

            self.quality_progression.append(self.best_quality)
            self._update_pheromone(all_paths, all_qualities)
            print(f"Iteration {iteration + 1}/{self.n_iterations}: Best Quality = {self.best_quality:.2f}")

        end_time = time.time()
        print(f"ACO optimization completed in {end_time - start_time:.2f} seconds.")

    def plot_quality_progression(self):
        """Plot the progression of the best quality score over iterations."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.n_iterations + 1), self.quality_progression, marker='o')
        plt.title("Best Recommendation Quality Progression")
        plt.xlabel("Iteration")
        plt.ylabel("Quality (Sum of Popularity Scores)")
        plt.grid(True)
        plt.show()

    def display_recommendations(self):
        """
        Display the best recommended songs using the best path found.
        Prints song details and returns the recommendation DataFrame.
        """
        recommendations = self.data.iloc[self.best_path]
        print("Best Recommendation Path (Song IDs):", self.best_path)
        print("Recommended Songs:")
        print(recommendations[['song_name', self.popularity_col]])
        return recommendations


# --------------------- Main Execution ---------------------
def main():
    # NOTE: Replace 'your_dataset.csv' with the actual path to your dataset.
    # The dataset should have at least the following columns:
    # 'song_name' (string): Name of the song.
    # 'popularity' (numeric): A popularity score of the song.
    # Additional feature columns (e.g., 'danceability', 'energy', 'tempo', etc.) for similarity calculations.
    dataset_path = 'your_dataset.csv'  # Update with your dataset path
    try:
        data = pd.read_csv(dataset_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print("Error loading dataset. Generating a dummy dataset for demonstration purposes.")
        # Generate a dummy dataset for demonstration
        n_dummy = 50
        data = pd.DataFrame({
            'song_name': [f"Song {i}" for i in range(n_dummy)],
            'popularity': np.random.randint(50, 100, n_dummy),
            'danceability': np.random.rand(n_dummy),
            'energy': np.random.rand(n_dummy),
            'tempo': np.random.rand(n_dummy) * 200
        })

    # List of feature columns to be used for similarity (update as needed)
    feature_cols = ['danceability', 'energy', 'tempo']
    popularity_col = 'popularity'

    # Initialize the ACO recommender with advanced parameters
    aco = ACORecommender(data=data, feature_cols=feature_cols, popularity_col=popularity_col,
                         n_ants=20, n_iterations=100, alpha=1.0, beta=3.0, rho=0.05, Q=1.0)

    # Run the optimization process
    aco.run_optimization()

    # Plot the quality progression over iterations
    aco.plot_quality_progression()

    # Display the best recommendations
    recommendations = aco.display_recommendations()

    # OPTIONAL: If you have ground truth for recommendations, calculate precision and recall.
    # For demonstration, we create dummy ground truth binary lists.
    # In practice, these should be replaced with actual labels.
    y_true = np.random.randint(0, 2, len(recommendations))
    y_pred = np.random.randint(0, 2, len(recommendations))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"Dummy Precision: {precision:.2f}, Dummy Recall: {recall:.2f}")


if __name__ == "__main__":
    main()
