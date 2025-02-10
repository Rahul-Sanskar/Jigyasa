from sklearn.linear_model import LinearRegression
import numpy as np

class AdaptiveLearningAI:
    def __init__(self):
        self.model = LinearRegression()
        self.data = []
        self.labels = []

    def update_data(self, new_data, difficulty_level):
        """ Stores past learning data """
        self.data.append([new_data])  # Ensuring it is a 2D array
        self.labels.append(difficulty_level)

    def train_model(self):
        """ Trains the model if data is available """
        if len(self.data) > 1:  # Need at least two points to train
            X = np.array(self.data)
            y = np.array(self.labels)
            self.model.fit(X, y)

    def recommend_difficulty(self):
        """ Predicts the next difficulty level based on learning progress """
        if len(self.data) < 1:
            return 1  # Default difficulty if no data available
        self.train_model()  # Ensure model is trained before predicting
        return max(1, int(self.model.predict([[len(self.data)]])[0]))


# Example Usage
adaptiveAI = AdaptiveLearningAI()

# Simulated learning progress (e.g., session number and difficulty level)
adaptiveAI.update_data(1, 1)
adaptiveAI.update_data(2, 2)
adaptiveAI.update_data(3, 3)

# Ensure the model is trained before prediction
adaptiveAI.train_model()

print("Recommended Difficulty Level:", adaptiveAI.recommend_difficulty())
