
# =====================================================
# CodeAlpha Internship - Task 2
# Unemployment Analysis with Python
# Dataset: Unemployment in India.csv
# =====================================================

# Step 1: Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset

df = pd.read_csv("Unemployment in India.csv")

print("\n========== DATASET PREVIEW ==========")
print(df.head())

# Step 3: Clean Column Names

df.columns = df.columns.str.strip()

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Step 4: Dataset Information

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

# Step 5: Remove Extra Spaces

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Step 6: Check Missing Values

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Step 7: Clean Dataset

df = df.dropna().copy()

# Step 8: Convert Date Column

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["Date"])

# Step 9: Statistical Analysis

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\nAverage Unemployment Rate:",
      round(df["Estimated Unemployment Rate (%)"].mean(), 2), "%")

print("\nMaximum Unemployment Rate:",
      round(df["Estimated Unemployment Rate (%)"].max(), 2), "%")

print("\nMinimum Unemployment Rate:",
      round(df["Estimated Unemployment Rate (%)"].min(), 2), "%")

# Step 10: Monthly Unemployment Trend

monthly_data = df.groupby(
    df["Date"].dt.to_period("M")
)["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12, 6))
monthly_data.plot(kind="line", marker="o")

plt.title("Monthly Unemployment Rate in India")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: State-wise Unemployment Analysis

region_data = df.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 7))
region_data.plot(kind="bar")

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Step 12: COVID-19 Impact Analysis

df["Period"] = df["Date"].apply(
    lambda date: "Before COVID-19"
    if date < pd.Timestamp("2020-03-01")
    else "During COVID-19"
)

covid_data = df.groupby("Period")[
    "Estimated Unemployment Rate (%)"
].mean()

print("\n========== COVID-19 IMPACT ==========")
print(covid_data)

plt.figure(figsize=(8, 5))
covid_data.plot(kind="bar")

plt.title("Unemployment Rate Before and During COVID-19")
plt.xlabel("Period")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 13: Correlation Heatmap

numeric_data = df.select_dtypes(include="number")

plt.figure(figsize=(9, 6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Variables")
plt.tight_layout()
plt.show()

# Step 14: Final Insights

print("\n========== FINAL PROJECT INSIGHTS ==========")

print("Total Records Analyzed:", len(df))

print("Highest Unemployment Region:", region_data.idxmax())

print("Lowest Unemployment Region:", region_data.idxmin())

print(
    "Highest Monthly Average Unemployment:",
    round(monthly_data.max(), 2), "%"
)

print("\n========== PROJECT COMPLETED ==========")
