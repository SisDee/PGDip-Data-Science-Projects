#!/usr/bin/env python
# coding: utf-8

# In[174]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# In[ ]:



   


# ### Data
# The datases used is the Breast Cancer Wisconsin (Diagnostic) Data Set. This dataset consist of 569 samples with 30 features that are related to diagnosig breast cancer. While this is a labelled dataset we drop any label data in order to use it to explore PCA and Kmeans and see if we can discover clusters ourselves.
# 

# In[175]:


df = pd.read_csv('data/data.csv')
# Remove labels
df = df.loc[:, ~df.columns.isin(['id', 'Unnamed: 32', 'diagnosis'])]
df


# In[ ]:





# In[176]:


df.describe()


# ### Data Processing
# 
# - Scale Features using Standard Scalar - This is important so that actual magnitude of a feature in terms of its variance etc does not come into effect when we do PCA later

# In[178]:


scaler = StandardScaler()
scaler.fit(df)
df_scaled = df.copy()
df_scaled.loc[:, :] = scaler.transform(df.loc[:, :])
df_scaled


# ### Principal Component Analysis
# All our features are now numerical and we can go directly to doing PCA

# In[197]:


pca = PCA(n_components = df_scaled.shape[1])
pca.fit(df_scaled)
variances = pca.explained_variance_ratio_
components = 1+np.arange(len(variances))


fig,axis = plt.subplots(1,2,figsize=(15,5))

axis[0].set(xlabel = "Principal Component",ylabel="Variance % explained",
            title = "Variance ratio by principal component")
axis[0].bar(components.astype(str),variances )
axis[0].grid()


cumulative_variances = np.cumsum(variances)

axis[1].set(xlabel = "Principal components",ylabel = "Percentage of variance retained",
            title = "Total variance retained from principal components")
axis[1].scatter(components,cumulative_variances * 100)
axis[1].plot(components,cumulative_variances * 100)
axis[1].grid()
min_components_for_90_variance = np.where(cumulative_variances >= 0.90)[0][0] + 1
print("Number of components for 90% variation ", min_components_for_90_variance)
print("Explained variance ratios ", pca.explained_variance_ratio_)
fig.tight_layout()


# In[199]:


i = 0
for p in pca.explained_variance_ratio_:
    print(f'Principal Component {i+1} : {p}')
    i = i+1


# In[188]:


pca_ = PCA(n_components = min_components_for_90_variance)
df_pca_space = pca_.fit_transform(df_scaled)


# ### Kmeans Clustering

# In[191]:


def k_means(model_supplier, data):
    inertia = []
    for n in range(1 , 11):
        model = model_supplier(n)
        model.fit(data)
        inertia.append(model.inertia_)
    return inertia


# In[192]:


model = lambda n: (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=100, 
                        tol=0.001,  random_state= 2, algorithm='elkan')) 
raw_data_inertia = k_means(model, df_encoded)
pca_data_inertia = k_means(model, df_pca_space)


# In[193]:


fig, ax = plt.subplots( 1, 2 , figsize = (10 ,4))
ax[0].plot(np.arange(1 , 11) , raw_data_inertia , 'o')
ax[0].plot(np.arange(1 , 11) , raw_data_inertia , '-' , alpha = 0.5)
ax[0].set_title('Elbow Curve in original Feature Space')
ax[0].set_xlabel('Number of Clusters')
ax[0].set_ylabel('Inertia (WCSS)')
ax[1].plot(np.arange(1 , 11) , pca_data_inertia , 'o')
ax[1].plot(np.arange(1 , 11) , pca_data_inertia , '-' , alpha = 0.5)
ax[1].set_xlabel('Number of Clusters')
ax[1].set_ylabel('Inertia (WCSS)')
ax[1].set_title('Elbow Curve in PCA Space')
fig.tight_layout()
ax[0].grid()
ax[1].grid()
plt.show()


# In[194]:


k = 2
optimal_model = model(k)
optimal_model.fit(df_pca_space)


# In[195]:


df_with_pca_comp = pd.concat([df, pd.DataFrame(df_pca_space)], axis = 1)
df_with_pca_comp.rename(
    columns = {i : f'Principal Component {i+1}' for i in range(0,min_components_for_90_variance)}, inplace = True)
df_with_pca_comp['Label'] = optimal_model.labels_
df_with_pca_comp


# In[196]:


components = [f'Principal Component {i+1}' for i in range(min_components_for_90_variance)] + ['Label']
sns.pairplot(df_with_pca_comp.loc[:, df_with_pca_comp.columns.isin(components)],
             hue = 'Label',diag_kind = 'kde',
             kind = 'scatter',
             palette = 'rainbow')


# From the above plots there are 2 clusters in the dataset . The separtion into these 2 clusters can be cleary seen in the plot of principal component 1 vs other principal components . Shows that ,ost variation captured by this component.  

# ### Notes for report
# 
# 
# #### Dataset 
# The datases used is the Breast Cancer Wisconsin (Diagnostic) Data Set. This dataset consist of 569 samples with 30 features that are related to diagnosig breast cancer. While this is a labelled dataset we drop any label data in order to use it to explore PCA and Kmeans and see if we can discover clusters ourselves.
# 
# 
#  
# 
# #### Processing
# Standard scaler on features. This is necessary since principal component analysis will yield new projection of the data that will be derived from standar deviation/variance of initial data so we don't want the actual magnitude of the standard deviation of a variable to influence calculation of new axis. 
# 
# #### Principal Component Analysis
# Performed PCA and components have the following explained variance ratios
# 
# - PC1 ~ 0.38
# - PC2 ~ 0.28
# - PC3 ~ 0.19
# - PC4 ~ 0.138
# - PC5  ~0 i.e too small ^-34
# 
# to keep 90% variation in the dataset we only need 4 principal components (first 4) so we can reduce dimensionality by 1 and still have lot of variation .
# 
# #### Kmeans Clustering
# Perform kmeans clustering on the projected data or in PCA space with 7 components. To determine optimal k value we do a parameter sweep over values in the range [1,11], the elbow kink occurs at 2. So we use 2 as number of clusters in the dataset. 
# 
# 
# #### Limitations
# While does reduce dimnsionality we lose interpretability i.e we discarded 23 components but do not necessarily know which features they correspond to in feature space
# 
# 
#   
# 

# In[ ]:




