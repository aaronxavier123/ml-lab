import numpy as np
import pandas as pd

class DecisionTree:
    def _init_(self, criterion='entropy'):
        self.criterion = criterion

    def entropy(self, y):
        """
        Calculate entropy of a given target variable
        """
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def information_gain(self, X, y, feature_idx):
        """
        Calculate information gain for a given feature
        """
        # Calculate parent entropy
        parent_entropy = self.entropy(y)

        # Calculate weighted sum of child entropies
        unique_values = np.unique(X.iloc[:, feature_idx])
        child_entropy = 0
        for value in unique_values:
            subset_indices = X.index[X.iloc[:, feature_idx] == value].tolist()
            subset_y = y[subset_indices]
            child_entropy += (len(subset_y) / len(y)) * self.entropy(subset_y)

        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def find_best_split(self, X, y):
        """
        Find the best feature to split on
        """
        best_feature_idx = None
        best_information_gain = -np.inf

        for feature_idx in range(X.shape[1]):
            ig = self.information_gain(X, y, feature_idx)
            if ig > best_information_gain:
                best_information_gain = ig
                best_feature_idx = feature_idx

        return best_feature_idx

    def fit(self, X, y):
        """
        Fit the Decision Tree model
        """
        self.root_node = self.find_best_split(X, y)

    def get_root_node(self):
        """
        Get the root node of the Decision Tree
        """
        return self.root_node

# Example usage
if _name_ == "_main_":
    # Load data from CSV file
    filepath=r"C:/Users/akhil/Downloads/DATA/DATA/code_comm.csv"
    data = pd.read_csv(filepath)  # Change 'your_dataset.csv' to the name of your CSV file
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target variable

    # Create DecisionTree instance
    dt = DecisionTree()

    # Fit the model
    dt.fit(X, y)

    # Get the root node
    root_node = dt.get_root_node()
    print("Root node index:", root_node)