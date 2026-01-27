# Linear Regression using Python (scikit-learn)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Create dataset (Study Hours vs Marks)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([10, 20, 30, 40, 50, 55, 65, 80])

print("Hours:", hours.flatten())
print("Marks:", marks)


# Step 2: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    hours, marks, test_size=0.2, random_state=42
)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))


# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Step 4: Make predictions
y_pred = model.predict(X_test)

print("\nActual Marks:", y_test)
print("Predicted Marks:", y_pred)


# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)


# Step 6: Predict for new value
new_hours = np.array([[9]])
prediction = model.predict(new_hours)

print("\nPrediction:")
print("If a student studies for 9 hours, predicted marks are:", round(prediction[0], 2))


# Step 7: Visualize the regression line
plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression: Study Hours vs Marks")
plt.show()
