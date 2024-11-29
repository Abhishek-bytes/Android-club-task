import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Customer Churn Prediction")
    
    uploaded_file = st.file_uploader("Upload marketing_campaign.csv", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            # Load dataset
            data = pd.read_csv(uploaded_file, sep='\t')
            st.write("Data Loaded Successfully!")
            st.write(data.head())
            
            # Data Cleaning
            data['Age'] = 2024 - data['Year_Birth']
            data['Age'] = data['Age'].fillna(data['Age'].median())
            data = data.drop(['ID', 'Year_Birth'], axis=1)
            
            # Fill missing values
            for column in data.columns:
                if data[column].isnull().sum() > 0:
                    data[column].fillna(data[column].mode()[0], inplace=True)
            
            # K-Means Clustering
            numeric_cols = data.select_dtypes(include=[np.number])
            kmeans = KMeans(n_clusters=3, random_state=42)
            numeric_cols['Cluster'] = kmeans.fit_predict(numeric_cols)
            
            # PCA Visualization
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(numeric_cols.drop('Cluster', axis=1))
            
            reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
            reduced_df['Cluster'] = numeric_cols['Cluster']
            
            # Visualize Clusters
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
            plt.title('Customer Clusters')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            st.pyplot(plt)
        
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.write("Upload a file to start analysis")

if __name__ == '__main__':
    main()
