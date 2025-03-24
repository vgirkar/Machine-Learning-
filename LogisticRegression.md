# Logistic Regression: An Overview  

Logistic regression is a fundamental statistical and machine learning algorithm used for binary classification problems. It predicts the probability that a given input belongs to one of two classes, making it widely used in applications like spam detection, medical diagnosis, and credit risk assessment. Unlike linear regression, which predicts continuous values, logistic regression estimates probabilities using the logistic (sigmoid) function to map predictions between 0 and 1.

## **Mathematical Foundation**  

Logistic regression models the relationship between independent variables (\(X\)) and a binary dependent variable (\(Y\)) using the sigmoid function:  

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \(z\) is a linear combination of input features:  

\[
z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n
\]

Here, \(w_0\) is the bias (intercept), \(w_1, w_2, ..., w_n\) are the weights (coefficients), and \(x_1, x_2, ..., x_n\) are the input features. The sigmoid function ensures the output is a probability value between 0 and 1. If the probability is greater than a chosen threshold (typically 0.5), the input is classified into one class; otherwise, it is assigned to the other class.

## **Loss Function and Optimization**  

Logistic regression uses the **log-likelihood function** as its loss function, given by the **binary cross-entropy** formula:

\[
L = - \frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i}) \right]
\]

where:  
- \(m\) is the number of training examples  
- \(y_i\) is the actual class label (0 or 1)  
- \(\hat{y_i}\) is the predicted probability for class 1  

The model's weights are optimized using **gradient descent**, which iteratively updates the weights to minimize the loss function.

## **Regularization**  

To prevent overfitting, logistic regression often includes **L1 (Lasso) or L2 (Ridge) regularization**. These add penalty terms to the loss function:  

- **L1 Regularization (Lasso)**: Adds \(\lambda \sum |w_i|\), promoting sparsity by forcing some weights to be exactly zero.  
- **L2 Regularization (Ridge)**: Adds \(\lambda \sum w_i^2\), reducing the magnitude of weights without forcing them to zero.  

## **Applications**  

1. **Medical Diagnosis** – Predicting diseases (e.g., diabetes, cancer detection).  
2. **Spam Detection** – Classifying emails as spam or not spam.  
3. **Credit Risk Assessment** – Determining loan eligibility based on financial history.  
4. **Marketing** – Predicting whether a customer will buy a product.  

## **Advantages and Limitations**  

### **Advantages:**  
- Simple and interpretable.  
- Efficient for small to medium-sized datasets.  
- Outputs calibrated probabilities useful in decision-making.  

### **Limitations:**  
- Assumes a linear relationship between input features and log-odds.  
- Not effective for complex datasets with non-linear decision boundaries.  
- Can be sensitive to outliers.  

Despite its simplicity, logistic regression remains a widely used baseline model for classification problems, providing a solid foundation for more complex algorithms.
