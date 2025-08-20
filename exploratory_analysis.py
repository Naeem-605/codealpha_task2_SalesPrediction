import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(df):
    """
    Perform exploratory data analysis
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Basic statistics
    print("Descriptive Statistics:")
    print(df.describe())
    
    # 2. Correlation analysis
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix of Advertising Data')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # 3. Pairplot to visualize relationships
    pairplot = sns.pairplot(df)
    pairplot.fig.suptitle('Pairplot of Advertising Data', y=1.02)
    plt.savefig('pairplot.png')
    plt.close()
    
    # 4. Distribution of sales
    plt.figure(figsize=(10, 6))
    plt.hist(df['Sales'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sales')
    plt.grid(True, alpha=0.3)
    plt.savefig('sales_distribution.png')
    plt.close()
    
    # 5. Advertising channels vs Sales
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channels = ['TV', 'Radio', 'Newspaper']
    for i, channel in enumerate(channels):
        axes[i].scatter(df[channel], df['Sales'], alpha=0.6)
        axes[i].set_xlabel(f'{channel} Advertising')
        axes[i].set_ylabel('Sales')
        axes[i].set_title(f'Sales vs {channel} Advertising')
        # Add trend line
        z = np.polyfit(df[channel], df['Sales'], 1)
        p = np.poly1d(z)
        axes[i].plot(df[channel], p(df[channel]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('advertising_vs_sales.png')
    plt.close()
    
    print("EDA visualizations saved as PNG files!")

if __name__ == "__main__":
    df = pd.read_csv('E:\Code alpha Internship\Sales Prediction using python\Advertising.csv', index_col=0)
    perform_eda(df)