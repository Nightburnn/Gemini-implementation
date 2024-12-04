import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # ==============================
    # 1. Exploratory Data Analysis (EDA) on Diabetes Dataset
    # ==============================
    # Load the Diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    # Add the target variable to the DataFrame
    df['target'] = diabetes.target

    print("====== Diabetes Dataset EDA ======")
    # Print basic information about the dataset
    print("Dataset Shape:", df.shape) # Dimensions of the dataset
    print("First 5 rows:\n", df.head())  # Display the first 5 rows
    # Generate descriptive statistics
    print("\nSummary Statistics:\n", df.describe())

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    # Plot a heatmap to visualize feature correlations
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # Correlation heatmap
    plt.title("Correlation Heatmap - Diabetes Dataset")
    plt.show()

    # ==============================
    # 2. Generate Synthetic Sample Data (Age, Height, Weight)
    # ==============================
    # Generate random data for age, height, and weight
    np.random.seed(42)  # Set seed for reproducibility
    age = np.random.randint(18, 60, 100)  # Random ages between 18 and 60
    height = np.random.normal(165, 10, 100)   # Height around 165 cm with std deviation of 10
    weight = np.random.normal(70, 15, 100) # Weight around 70 kg with std deviation of 15

    # Create a DataFrame to store the data
    synthetic_data = pd.DataFrame({'Age': age, 'Height': height, 'Weight': weight})
# Print the first 5 rows of the synthetic dataset same parttern as the diabetes dataset
    print("\n====== Synthetic Data ======")
    print("First 5 rows of synthetic data:\n", synthetic_data.head())

    # ==============================
    # 3. Logistic Regression on Iris Dataset
    # ==============================
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target

# Split the data into training and testing sets
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

# Evaluate the model
    print("\n====== Logistic Regression on Iris Dataset ======")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
