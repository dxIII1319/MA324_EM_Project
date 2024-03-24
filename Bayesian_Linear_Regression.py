import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def compute_posterior(X, Y, alpha, beta):
    # Compute posterior mean and covariance
    Sigma_theta_inv = alpha * np.eye(X.shape[1]) + beta * np.dot(X.T, X)
    Sigma_theta = np.linalg.inv(Sigma_theta_inv)
    mu_theta = beta * np.dot(Sigma_theta, np.dot(X.T, Y))
    return mu_theta, Sigma_theta

def update_theta(X, Y, alpha, beta, mu_theta, Sigma_theta):
    # Update theta using posterior mean and covariance
    theta = np.random.multivariate_normal(mu_theta, Sigma_theta)
    return theta

def em_bayesian_linear_regression(X, Y, alpha, beta, epsilon=1e-6, max_iter=1000):
    # Initialization
    theta = np.zeros(X.shape[1])  # Initialize theta
    converged = False
    iter_count = 0
    
    # EM Iteration
    while not converged and iter_count < max_iter:
        # E-step
        mu_theta, Sigma_theta = compute_posterior(X, Y, alpha, beta)
        
        # M-step
        new_theta = update_theta(X, Y, alpha, beta, mu_theta, Sigma_theta)
        
        # Check for convergence
        if np.linalg.norm(new_theta - theta) < epsilon:
            converged = True
        else:
            theta = new_theta
            iter_count += 1
            
    return theta

# Generate synthetic data
np.random.seed(0)
N = 100  # Number of samples
D = 2    # Number of features

# True regression coefficients
true_theta = np.array([2.5, -1.0])

# Generate input matrix X
X = np.random.randn(N, D)

# Generate output vector Y
Y = np.dot(X, true_theta) + np.random.normal(0, 0.5, N)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameters
alpha = 1.0
beta = 1.0

# Perform Bayesian linear regression using EM algorithm
theta = em_bayesian_linear_regression(X_train, Y_train, alpha, beta)

# Predictions on test set
Y_pred = np.dot(X_test, theta)

# Evaluate performance
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Simple Linear Regression
simple_model = LinearRegression()
simple_model.fit(X_train, Y_train)
Y_pred_simple = simple_model.predict(X_test)

# Evaluate performance of simple linear regression
mse_simple = mean_squared_error(Y_test, Y_pred_simple)
r2_simple = r2_score(Y_test, Y_pred_simple)

# Compare performance
print("Bayesian Linear Regression - Mean Squared Error (MSE):", mse)
print("Simple Linear Regression - Mean Squared Error (MSE):", mse_simple)
print("Bayesian Linear Regression - R-squared (R2):", r2)
print("Simple Linear Regression - R-squared (R2):", r2_simple)