
# coding: utf-8

# In[149]:


import pandas as pd

from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

data_df = pd.read_csv('Banknote-authentication-dataset.csv')

#print(data)
des = data_df.describe()
print(des)

v1 = data_df['V1']
v2 = data_df['V2']


v1_v2 = np.column_stack( (v1 , v2) )


km = KMeans(n_clusters =2).fit(v1_v2)

data_df['KMeans'] = km_res.labels_

#print(km)

clusters_km = km.cluster_centers_

g = sb.FacetGrid(data = data_df, hue = 'KMeans', height = 8)
g.map(plt.scatter, 'V1', 'V2')
g.add_legend();
plt.scatter(clusters[:,0], clusters[:,1], s=500, marker='*', c='r')



#plt.scatter( v1, v2)
#plt.scatter(clusters_km[:,0],clusters_km[:,1], s=300, color ='red')

#plt.xlabel('v1')
#plt.ylabel('v2')

print(clusters_km)

data_df['KMeans'] = km_res.labels_

data_df.groupby('KMeans').describe()


# In[145]:


data_norm = ( data - data.min() ) / ( data.max() - data.min() )

#print (data_norm)
#desc = data_norm.describe()
#desc
#print(desc)

v1 = data['V1']
v2 = data['V2']

v1_v2 = np.column_stack( (v1 , v2) )

km_res = KMeans(n_clusters =2).fit(v1_v2)
clusters = km_res.cluster_centers_


plt.scatter( v1, v2, alpha =0.3, color ='blue')
plt.xlabel('v1')
plt.ylabel('v2')
#plt.scatter(clusters[:,0],clusters[:,1], s=300, color='orange')

#print(clusters)


# In[140]:


v1_norm = data_norm['V1']
v2_norm = data_norm['V2']

v1_v2_norm = np.column_stack( (v1_norm,v2_norm) )

km_res = KMeans(n_clusters =2).fit(v1_v2_norm)
clusters_norm = km_res.cluster_centers_



plt.scatter( data_norm['V1'], data_norm['V2'], cmap='summer')

plt.scatter(clusters[:,0],clusters[:,1], s=300, color='orange')

