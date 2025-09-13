# MachineLearning

## Description
This project demonstrates a simple linear regression analysis using two approaches:

1. Generating synthetic linear data with Gaussian noise.  
2. Solving linear regression using the **closed-form solution** (Normal Equation).  
3. Solving linear regression using **Gradient Descent**.  
4. Comparing both solutions with plots and a loss curve.


# 1. Import Libraries
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt

**2.Generate Synthetic Dataset & Plot Raw Data**
np.random.seed(42)             # For reproducibility
n_samples = 200                # Number of data points
X = np.random.uniform(0, 5, n_samples)   # Random x values between 0 and 5
noise = np.random.normal(0, 1, n_samples)  # Gaussian noise
y = 3 + 4 * X + noise          # True relationship with noise

# Plot the raw data
plt.scatter(X, y, color="royalblue", s=20, label="Data")
plt.title("Synthetic Dataset")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

**Closed-Form Solution (Normal Equation)**
X_bias = np.c_[np.ones((n_samples, 1)), X]  # Add bias (intercept) column
theta_normal = np.linalg.inv(X_bias.T @ X_bias) @ (X_bias.T @ y)
b_normal, w_normal = theta_normal

print(f"Closed-form coefficients -> Intercept: {b_normal:.4f}, Slope: {w_normal:.4f}")

**Plot Closed-Form Fitted Line**
plt.scatter(X, y, color="royalblue", s=20, label="Data")
plt.plot(X, X_bias @ theta_normal, color="crimson", lw=2, label="Normal Equation fit")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Fit via Closed-Form Solution")
plt.legend()
plt.show()

**Gradient Descent Implementation**
theta = np.zeros(2)       # Start with zeros
lr = 0.05                 # Learning rate
epochs = 1000             # Number of iterations
loss_values = []

for _ in range(epochs):
    preds = X_bias @ theta
    error = preds - y
    grad = (2 / n_samples) * X_bias.T @ error
    theta -= lr * grad
    loss_values.append(np.mean(error ** 2))

b_gd, w_gd = theta
print(f"Gradient Descent coefficients -> Intercept: {b_gd:.4f}, Slope: {w_gd:.4f}")

**Plot Loss Curve**
plt.plot(range(epochs), loss_values, color="seagreen")
plt.title("Gradient Descent: MSE vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.show()

**Comparison Plot**
plt.scatter(X, y, color="royalblue", s=20, label="Data")
plt.plot(X, X_bias @ theta_normal, color="crimson", lw=2, label="Normal Equation fit")
plt.plot(X, X_bias @ theta, color="orange", lw=2, ls="--", label="Gradient Descent fit")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Closed-Form vs Gradient Descent")
plt.legend()
plt.show()

**Short Explanation**
print("\nSummary:")
print(f"Normal Equation -> Intercept: {b_normal:.2f}, Slope: {w_normal:.2f}")
print(f"Gradient Descent -> Intercept: {b_gd:.2f}, Slope: {w_gd:.2f}")
print("Observation: With an appropriate learning rate, Gradient Descent converges to the same solution as the Closed-form approach.")
