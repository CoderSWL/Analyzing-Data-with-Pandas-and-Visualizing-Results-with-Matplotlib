# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from sklearn (you can replace this with any CSV dataset)
    iris = load_iris()
    
    # Convert the dataset to a DataFrame for easier manipulation
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())
    
    # Check data types and any missing values
    print("\nData types and missing values:")
    print(df.info())
    
    # Clean the dataset (if necessary)
    # In this case, the dataset does not have missing values, but let's ensure we handle them if they exist.
    df = df.dropna()  # Drop rows with missing values (if any)

except FileNotFoundError as e:
    print("Error: File not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
# Compute basic statistics of the numerical columns
print("\nBasic Statistics (mean, median, standard deviation, etc.):")
print(df.describe())

# Grouping by species and computing the mean of numerical columns
grouped = df.groupby('species').mean()
print("\nMean of numerical columns grouped by species:")
print(grouped)

# Task 3: Data Visualization
# Set up the figure for all plots
plt.figure(figsize=(14, 10))

# 1. Line chart: Showing trends over time (Simulating with a time-series for demonstration)
plt.subplot(2, 2, 1)
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title('Sepal Length Over Time')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# 2. Bar chart: Average petal length per species
plt.subplot(2, 2, 2)
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')

# 3. Histogram: Distribution of sepal width
plt.subplot(2, 2, 3)
plt.hist(df['sepal width (cm)'], bins=20, color='green', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# 4. Scatter plot: Relationship between sepal length and petal length
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df, hue='species', palette='Set1')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# Show all plots
plt.tight_layout()
plt.show()

# Observations or Findings
# From the visualizations and data analysis, we can observe:
# - Sepal length varies across species, with Iris-setosa having the shortest.
# - Petal length also varies significantly across species.
# - Sepal width shows a roughly normal distribution.
# - There is a clear correlation between sepal length and petal length, especially for Iris-virginica and Iris-setosa.
