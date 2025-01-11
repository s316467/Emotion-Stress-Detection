import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed ECG dataset
ecg_data = pd.read_csv("ECG.csv")

# Set output path for saving plots
output_path = "./"

# Data Exploration
print("Dataset shape:", ecg_data.shape)
print("Missing values:\n", ecg_data.isnull().sum())
print("Dataset preview:\n", ecg_data.head())

# 1. Plot feature distributions
def plot_feature_distributions(data, output_path):
    print("Plotting feature distributions...")
    for column in data.columns[1:]:  # Skip index if applicable
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{output_path}distribution_{column}.png")
        plt.close()

plot_feature_distributions(ecg_data, output_path)

# 2. Correlation Heatmap
def plot_correlation_heatmap(data, output_path):
    print("Plotting correlation heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_path}correlation_heatmap.png")
    plt.close()

plot_correlation_heatmap(ecg_data, output_path)

# 3. Time-Series Plot (if applicable)
def plot_time_series(data, output_path):
    print("Plotting time-series plots...")
    for column in data.columns[1:]:  # Skip index if applicable
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[column], color="green", label=column)
        plt.title(f"Time Series of {column}")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}timeseries_{column}.png")
        plt.close()

plot_time_series(ecg_data, output_path)

# 4. Boxplot for feature variability
def plot_feature_boxplot(data, output_path):
    print("Plotting feature boxplots...")
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=data.iloc[:, 1:])  # Skip index if applicable
    plt.xticks(rotation=90)
    plt.title("Boxplot of Features")
    plt.tight_layout()
    plt.savefig(f"{output_path}feature_boxplot.png")
    plt.close()

plot_feature_boxplot(ecg_data, output_path)

print("All plots have been saved to the folder.")
