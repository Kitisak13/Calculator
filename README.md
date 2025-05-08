# Logistic Regression in Machine Learning

Logistic Regression is a statistical method used in machine learning for binary classification problems. It predicts the probability that a given input belongs to a particular category. Despite its name, it is a classification algorithm, not a regression algorithm.

## Key Concepts

1. **Sigmoid Function**: Logistic regression uses the sigmoid function to map predicted values to probabilities between 0 and 1.
    \[
    \sigma(z) = \frac{1}{1 + e^{-z}}
    \]
2. **Decision Boundary**: A threshold (commonly 0.5) is applied to the probability to classify the input into one of the two categories.
3. **Cost Function**: The algorithm minimizes the log-loss function to optimize the model.

## Applications

- Spam email detection
- Medical diagnosis (e.g., predicting disease presence)
- Customer churn prediction

Logistic Regression is simple yet powerful for linearly separable data and serves as a foundation for more complex models.