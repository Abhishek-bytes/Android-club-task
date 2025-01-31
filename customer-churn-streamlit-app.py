import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # App title
    st.title("Customer Churn Prediction for Subscription-Based Business")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your marketing_campaign.csv file", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            # Step 1: Load dataset
            data = pd.read_csv(uploaded_file, sep='\t')
            st.write("### Data Loaded Successfully!")
            st.write("Here are the first few rows of the data:")
            st.write(data.head())
            
            # Step 2: Data Cleaning
            st.write("### Data Cleaning")
            data['Age'] = 2024 - data['Year_Birth']
            median_age = data['Age'].median()
            data['Age'] = data['Age'].fillna(median_age)
            
            # Drop unnecessary columns
            data = data.drop(['ID', 'Year_Birth'], axis=1)
            
            # Fill missing values with the mode
            for column in data.columns:
                if data[column].isnull().sum() > 0:
                    data[column].fillna(data[column].mode()[0], inplace=True)
            
            st.write("Data after cleaning:")
            st.write(data.head())
            
            # Step 3: K-Means Clustering
            st.write("### K-Means Clustering")
            numeric_cols = data.select_dtypes(include=[np.number])
            st.write("Selected numeric columns for clustering:")
            st.write(numeric_cols.head())
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=3, random_state=42)
            numeric_cols['Cluster'] = kmeans.fit_predict(numeric_cols)
            
            st.write("Data with clusters assigned:")
            st.write(numeric_cols.head())
            
            # Step 4: PCA for Visualization
            st.write("### PCA for Dimensionality Reduction")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(numeric_cols.drop('Cluster', axis=1))
            
            reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
            reduced_df['Cluster'] = numeric_cols['Cluster']
            
            st.write("PCA Results:")
            st.write(reduced_df.head())
            
            # Step 5: Visualization
            st.write("### Cluster Visualization")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
            plt.title("Clusters in Reduced Dimensional Space")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            st.pyplot(plt)
        
        except Exception as e:
            st.error(f"An error occurred while processing the data: {e}")
    else:
        st.write("Please upload your file to start the analysis.")

# Run the Streamlit app
if __name__ == '__main__':
    main()
