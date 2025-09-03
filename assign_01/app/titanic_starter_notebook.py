# Titanic Pandas Assignment - Learning-Focused Implementation
# This notebook is structured to help you understand WHY we use each pandas operation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# =============================================================================
# PART A - BASIC: Building Foundation
# =============================================================================

print("\n" + "="*50)
print("PART A - BASIC: Foundation Building")
print("="*50)

# Task 1: Load & Inspect
print("\n📊 Task 1: Load & Inspect")
print("-" * 30)

# LEARNING POINT: Why we specify dtype and na_values
# - Memory optimization: int32 vs int64 can save 50% memory
# - Consistent missing value handling across different data sources
try:
    df = pd.read_csv('../data_titanic/train.csv')
    print("✅ Data loaded successfully!")
    
    print(f"\n🔍 Dataset Shape: {df.shape}")
    print(f"   - Rows (samples): {df.shape[0]}")
    print(f"   - Columns (features): {df.shape[1]}")
    
    print(f"\n📋 Dataset Info:")
    df.info()
    
    print(f"\n👀 First 5 rows:")
    print(df.head())
    
    # LEARNING POINT: Memory usage analysis
    memory_usage = df.memory_usage(deep=True)
    print(f"\n💾 Memory Usage: {memory_usage.sum() / 1024**2:.2f} MB")
    
except FileNotFoundError:
    print("❌ Error: Could not find train.csv in ../data_titanic/")
    print("Make sure your data file is in the correct location!")

# Task 2: Column Summary
print("\n📊 Task 2: Column Summary")
print("-" * 30)

# LEARNING POINT: Programmatic data profiling
# This creates a reusable function for any dataset
def create_column_summary(dataframe):
    """
    Create comprehensive column summary for data profiling.
    
    Why this approach:
    - Scalable to any dataset size
    - Reveals data quality issues
    - Foundation for automated data quality reports
    """
    summary_data = []
    
    for col in dataframe.columns:
        col_info = {
            'column_name': col,
            'dtype': str(dataframe[col].dtype),
            'missing_count': dataframe[col].isnull().sum(),
            'missing_percentage': round(dataframe[col].isnull().sum() / len(dataframe) * 100, 2),
            'unique_values': dataframe[col].nunique(),
            'unique_percentage': round(dataframe[col].nunique() / len(dataframe) * 100, 2)
        }
        summary_data.append(col_info)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values('missing_count', ascending=False)

# Apply our custom function
column_summary = create_column_summary(df)
print("📋 Column Summary (sorted by missing values):")
print(column_summary)

# LEARNING POINT: What this tells us about data quality
print(f"\n🎯 Data Quality Insights:")
high_missing = column_summary[column_summary['missing_percentage'] > 50]
if not high_missing.empty:
    print(f"   - High missing data (>50%): {list(high_missing['column_name'])}")

categorical_cols = column_summary[
    (column_summary['dtype'] == 'object') & 
    (column_summary['unique_percentage'] < 10)
]['column_name'].tolist()
print(f"   - Likely categorical columns: {categorical_cols}")

# Task 3: Value Counts & Proportions
print("\n📊 Task 3: Value Counts & Proportions")
print("-" * 30)

# LEARNING POINT: Why analyze categorical distributions
categorical_columns = ['Pclass', 'Sex', 'Embarked']

for col in categorical_columns:
    print(f"\n🏷️  {col} Distribution:")
    
    # Value counts with percentages
    value_counts = df[col].value_counts()
    percentages = df[col].value_counts(normalize=True) * 100
    
    # Combine into a nice DataFrame
    distribution = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentages.round(2)
    })
    
    print(distribution)
    
    # LEARNING POINT: Check for data quality issues
    if df[col].isnull().sum() > 0:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        print(f"   ⚠️  Missing values: {df[col].isnull().sum()} ({missing_pct:.1f}%)")

# Task 4: Select & Filter
print("\n📊 Task 4: Select & Filter")
print("-" * 30)

# LEARNING POINT: Multiple approaches to filtering
print("🎯 Finding: Female passengers in 1st class older than 30")

# Approach 1: Step-by-step boolean indexing (easier to debug)
print("\n📋 Method 1: Step-by-step boolean indexing")
female_mask = df['Sex'] == 'female'
first_class_mask = df['Pclass'] == 1
age_mask = df['Age'] > 30

print(f"   - Female passengers: {female_mask.sum()}")
print(f"   - First class passengers: {first_class_mask.sum()}")
print(f"   - Passengers over 30: {age_mask.sum()}")

# Combine conditions
combined_mask = female_mask & first_class_mask & age_mask
female_firstclass_over_30 = df[combined_mask]

print(f"   - Combined conditions: {combined_mask.sum()}")

# Approach 2: Query method (more readable for complex conditions)
print("\n📋 Method 2: Using .query() method")
female_firstclass_over_30_query = df.query("Sex == 'female' and Pclass == 1 and Age > 30")

print(f"   - Query result count: {len(female_firstclass_over_30_query)}")
print(f"   - Results match: {len(female_firstclass_over_30) == len(female_firstclass_over_30_query)}")

# Sort by fare and show top 10
result = female_firstclass_over_30.sort_values('Fare', ascending=False).head(10)
print(f"\n🎖️  Top 10 by Fare:")
print(result[['Name', 'Age', 'Fare', 'Survived']].to_string())

# Task 5: Basic Aggregations
print("\n📊 Task 5: Basic Aggregations")
print("-" * 30)

# LEARNING POINT: Different measures of central tendency
print("📊 Age Statistics (handling missing values):")
age_stats = {
    'Mean': df['Age'].mean(),
    'Median': df['Age'].median(),
    'Mode': df['Age'].mode().iloc[0] if not df['Age'].mode().empty else 'No mode',
    'Missing Count': df['Age'].isnull().sum(),
    'Missing Percentage': df['Age'].isnull().sum() / len(df) * 100
}

for stat, value in age_stats.items():
    if isinstance(value, float):
        print(f"   - {stat}: {value:.2f}")
    else:
        print(f"   - {stat}: {value}")

print("\n💰 Mean Fare by Class:")
fare_by_class = df.groupby('Pclass')['Fare'].mean().round(2)
print(fare_by_class.to_string())

# LEARNING POINT: Survival analysis basics
print("\n🆘 Survival Rates:")
overall_survival = df['Survived'].mean()
print(f"   - Overall survival rate: {overall_survival:.1%}")

survival_by_gender = df.groupby('Sex')['Survived'].mean()
print(f"   - Survival by gender:")
for gender, rate in survival_by_gender.items():
    print(f"     * {gender.capitalize()}: {rate:.1%}")

# LEARNING POINT: Statistical significance of differences
print(f"\n📊 Gender survival difference: {survival_by_gender['female'] - survival_by_gender['male']:.1%}")

print("\n" + "="*50)
print("PART A COMPLETE - Foundation established!")
print("="*50)
print("\n🎓 What you learned in Part A:")
print("   ✅ How to load and inspect data systematically")
print("   ✅ Create reusable data profiling functions")
print("   ✅ Analyze categorical distributions")
print("   ✅ Use boolean indexing and query methods")
print("   ✅ Calculate basic statistics while handling missing data")
print("   ✅ Perform group-based analysis")

# =============================================================================
# PART B - INTERMEDIATE: Data Transformation Skills
# =============================================================================

print("\n" + "="*50)
print("PART B - INTERMEDIATE: Data Transformation")
print("="*50)

# Continue with Part B tasks...
# [The rest would continue with similar detailed explanations for each task]

print("\n🚀 Ready to continue with Part B!")
print("Each section builds on previous concepts while introducing new pandas functionality.")