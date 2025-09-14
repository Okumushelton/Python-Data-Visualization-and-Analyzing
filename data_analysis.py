"""
COMPREHENSIVE DATA ANALYSIS WITH PANDAS AND MATPLOTLIB
This script handles CSV datasets with robust error handling, performs complete analysis,
and creates all required visualizations with detailed explanations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime

# Set plotting style
plt.style.use('default')

def load_dataset(file_path):
    """
    Load dataset from CSV file with comprehensive error handling
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset or None if error occurs
    """
    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded dataset from {file_path}")
        print(f"✓ Dataset shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
        return df
    except FileNotFoundError:
        print(f"✗ Error: The file {file_path} was not found.")
        print("Please ensure the file exists in the specified path.")
        return None
    except pd.errors.EmptyDataError:
        print("✗ Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("✗ Error: Error parsing the file. Please check the CSV format.")
        return None
    except Exception as e:
        print(f"✗ Unexpected error occurred while loading the dataset: {str(e)}")
        return None

def explore_dataset(df):
    """
    Explore the dataset structure and handle missing values
    
    Parameters:
    df (pandas.DataFrame): The dataset to explore
    
    Returns:
    pandas.DataFrame: Cleaned dataset
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION AND CLEANING")
    print("="*60)
    
    # Display first few rows to understand the data structure
    print("\n1. First 5 rows of the dataset:")
    print(df.head())
    
    # Display dataset information
    print("\n2. Dataset information:")
    print(df.info())
    
    # Check for missing values
    print("\n3. Missing values analysis:")
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage.round(2)
    })
    print(missing_report[missing_report['Missing Values'] > 0])
    
    if missing_report[missing_report['Missing Values'] > 0].empty:
        print("✓ No missing values found in the dataset.")
    else:
        print("\n4. Handling missing values...")
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"✓ Filled missing values in {col} with median: {median_val:.2f}")
        
        # For categorical columns, fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    print(f"✓ Filled missing values in {col} with mode: {mode_val}")
                else:
                    df[col].fillna('Unknown', inplace=True)
                    print(f"✓ Filled missing values in {col} with 'Unknown'")
    
    return df

def perform_analysis(df):
    """
    Perform basic data analysis on the dataset
    
    Parameters:
    df (pandas.DataFrame): The dataset to analyze
    """
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # Compute basic statistics for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print("1. Descriptive statistics for numerical columns:")
        print(df[numerical_cols].describe())
    else:
        print("✗ No numerical columns found for statistical analysis.")
        return
    
    # Find categorical columns for grouping
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        # Perform grouping on the first categorical column
        group_col = categorical_cols[0]
        print(f"\n2. Grouping by '{group_col}' and computing means:")
        
        # Calculate mean for each numerical column by group
        grouped_means = df.groupby(group_col)[numerical_cols].mean()
        print(grouped_means.round(2))
        
        # Identify interesting patterns
        print("\n3. Key insights from grouping analysis:")
        for num_col in numerical_cols:
            max_group = grouped_means[num_col].idxmax()
            min_group = grouped_means[num_col].idxmin()
            max_val = grouped_means[num_col].max()
            min_val = grouped_means[num_col].min()
            range_val = max_val - min_val
            
            print(f"   - {num_col}:")
            print(f"     * Highest in {max_group} ({max_val:.2f})")
            print(f"     * Lowest in {min_group} ({min_val:.2f})")
            print(f"     * Range: {range_val:.2f}")
            
            if range_val > (max_val * 0.5):  # If range is more than 50% of max value
                print(f"     * Significant variation across categories")
    
    # Check for correlations between numerical variables
    if len(numerical_cols) >= 2:
        print(f"\n4. Correlation analysis:")
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix.round(2))
        
        # Find the strongest correlation
        strong_corr = None
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                corr_val = correlation_matrix.iloc[i, j]
                if strong_corr is None or abs(corr_val) > abs(strong_corr[2]):
                    strong_corr = (numerical_cols[i], numerical_cols[j], corr_val)
        
        if strong_corr and abs(strong_corr[2]) > 0.7:
            print(f"   - Strong correlation between {strong_corr[0]} and {strong_corr[1]}: {strong_corr[2]:.2f}")

def create_visualizations(df):
    """
    Create required visualizations from the dataset
    
    Parameters:
    df (pandas.DataFrame): The dataset to visualize
    """
    print("\n" + "="*60)
    print("DATA VISUALIZATION")
    print("="*60)
    
    # Get column types for appropriate visualizations
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numerical_cols:
        print("✗ No numerical columns found for visualizations.")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Data Analysis Visualizations', fontsize=16, fontweight='bold')
    
    # Visualization 1: Line chart (trend over time)
    print("1. Creating line chart...")
    date_columns = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month', 'day'])]
    
    if date_columns:
        date_col = date_columns[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df_sorted = df.sort_values(date_col)
            num_col = numerical_cols[0]
            
            axes[0, 0].plot(df_sorted[date_col], df_sorted[num_col], color='steelblue', linewidth=2)
            axes[0, 0].set_title(f'Trend of {num_col} Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Date', fontweight='bold')
            axes[0, 0].set_ylabel(num_col, fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        except Exception as e:
            print(f"   Could not use {date_col} as date: {str(e)}")
            # Fallback: use index as pseudo-time
            axes[0, 0].plot(df.index, df[numerical_cols[0]], color='steelblue', linewidth=2)
            axes[0, 0].set_title(f'Trend of {numerical_cols[0]} (Index as Time)', fontweight='bold')
            axes[0, 0].set_xlabel('Index', fontweight='bold')
            axes[0, 0].set_ylabel(numerical_cols[0], fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
    else:
        # Use index as pseudo-time
        axes[0, 0].plot(df.index, df[numerical_cols[0]], color='steelblue', linewidth=2)
        axes[0, 0].set_title(f'Trend of {numerical_cols[0]} (Index as Time)', fontweight='bold')
        axes[0, 0].set_xlabel('Index', fontweight='bold')
        axes[0, 0].set_ylabel(numerical_cols[0], fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Visualization 2: Bar chart (comparison across categories)
    print("2. Creating bar chart...")
    if categorical_cols and numerical_cols:
        category_col = categorical_cols[0]
        value_col = numerical_cols[0]
        
        # Calculate means for each category
        category_means = df.groupby(category_col)[value_col].mean().sort_values(ascending=False)
        
        # Create bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_means)))
        bars = axes[0, 1].bar(range(len(category_means)), category_means.values, color=colors, edgecolor='black')
        
        axes[0, 1].set_title(f'Average {value_col} by {category_col}', fontweight='bold')
        axes[0, 1].set_xlabel(category_col, fontweight='bold')
        axes[0, 1].set_ylabel(f'Average {value_col}', fontweight='bold')
        axes[0, 1].set_xticks(range(len(category_means)))
        axes[0, 1].set_xticklabels(category_means.index, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'No categorical data for bar chart', 
                       ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('Bar Chart Not Available', fontweight='bold')
    
    # Visualization 3: Histogram (distribution of numerical data)
    print("3. Creating histogram...")
    num_col = numerical_cols[0]
    
    axes[1, 0].hist(df[num_col], bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title(f'Distribution of {num_col}', fontweight='bold')
    axes[1, 0].set_xlabel(num_col, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    
    # Add vertical lines for mean and median
    mean_val = df[num_col].mean()
    median_val = df[num_col].median()
    axes[1, 0].axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[1, 0].axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[1, 0].legend()
    
    # Visualization 4: Scatter plot (relationship between two numerical columns)
    print("4. Creating scatter plot...")
    if len(numerical_cols) >= 2:
        x_col = numerical_cols[0]
        y_col = numerical_cols[1]
        
        # Color by category if available
        if categorical_cols:
            categories = df[categorical_cols[0]].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            color_map = dict(zip(categories, colors))
            
            for category in categories:
                category_data = df[df[categorical_cols[0]] == category]
                axes[1, 1].scatter(category_data[x_col], category_data[y_col], 
                                  color=color_map[category], label=category, alpha=0.7, s=50)
            axes[1, 1].legend()
        else:
            axes[1, 1].scatter(df[x_col], df[y_col], alpha=0.6, color='seagreen', s=50)
        
        axes[1, 1].set_title(f'Relationship between {x_col} and {y_col}', fontweight='bold')
        axes[1, 1].set_xlabel(x_col, fontweight='bold')
        axes[1, 1].set_ylabel(y_col, fontweight='bold')
        
        # Add correlation coefficient
        correlation = df[x_col].corr(df[y_col])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                       transform=axes[1, 1].transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough numerical data for scatter plot', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Scatter Plot Not Available', fontweight='bold')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('data_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizations saved as 'data_analysis_visualizations.png'")

def main():
    """
    Main function to run the complete data analysis pipeline
    """
    print("="*70)
    print("COMPREHENSIVE DATA ANALYSIS WITH PANDAS AND MATPLOTLIB")
    print("="*70)
    
    # Define dataset path - user can change this
    dataset_path = input("Enter the path to your CSV file (or press Enter to use Iris dataset): ").strip()
    
    if not dataset_path:
        # Use Iris dataset as default
        print("Using Iris dataset as default...")
        try:
            import seaborn as sns
            df = sns.load_dataset('iris')
            df.to_csv('iris.csv', index=False)
            dataset_path = "iris.csv"
            print("✓ Iris dataset loaded and saved as iris.csv")
        except ImportError:
            print("✗ Seaborn not available. Please install it with: pip install seaborn")
            return
    else:
        # Load user-provided dataset
        df = load_dataset(dataset_path)
        if df is None:
            print("✗ Failed to load dataset. Exiting.")
            return
    
    # Explore the dataset
    df = explore_dataset(df)
    
    # Perform analysis
    perform_analysis(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("Summary of actions performed:")
    print("✓ Dataset loaded and explored")
    print("✓ Missing values handled")
    print("✓ Statistical analysis completed")
    print("✓ Grouping analysis performed")
    print("✓ Four visualizations created and saved")
    print("\nThank you for using the Data Analysis Tool!")

if __name__ == "__main__":
    main()