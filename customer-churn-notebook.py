{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Customer Churn Prediction Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.decomposition import PCA\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "# Note: Replace 'marketing_campaign.csv' with your actual file path\n",
    "data = pd.read_csv('marketing_campaign.csv', sep='\\t')\n",
    "print(\"Dataset loaded successfully!\")\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Cleaning\n",
    "# Calculate Age\n",
    "data['Age'] = 2024 - data['Year_Birth']\n",
    "\n",
    "# Handle missing values\n",
    "data['Age'] = data['Age'].fillna(data['Age'].median())\n",
    "\n",
    "# Drop unnecessary columns\n",
    "data = data.drop(['ID', 'Year_Birth'], axis=1)\n",
    "\n",
    "# Fill missing values in other columns\n",
    "for column in data.columns:\n",
    "    if data[column].isnull().sum() > 0:\n",
    "        data[column].fillna(data[column].mode()[0], inplace=True)\n",
    "\n",
    "print(\"Data cleaning completed\")\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare numeric columns for clustering\n",
    "numeric_cols = data.select_dtypes(include=[np.number])\n",
    "\n",
    "# Apply K-Means Clustering\n",
    "kmeans = KMeans(n_clusters=3, random_state=42)\n",
    "numeric_cols['Cluster'] = kmeans.fit_predict(numeric_cols)\n",
    "\n",
    "print(\"Clustering completed\")\n",
    "numeric_cols.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Perform PCA for visualization\n",
    "pca = PCA(n_components=2)\n",
    "reduced_data = pca.fit_transform(numeric_cols.drop('Cluster', axis=1))\n",
    "\n",
    "reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])\n",
    "reduced_df['Cluster'] = numeric_cols['Cluster']\n",
    "\n",
    "print(\"PCA transformation completed\")\n",
    "reduced_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize Clusters\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='viridis')\n",
    "plt.title('Customer Clusters')\n",
    "plt.xlabel('Principal Component 1')\n",
    "plt.ylabel('Principal Component 2')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
