#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the various libraries

import folium
import numpy as np
import pandas as pd
import json
from geopy.geocoders import Nominatim
from pandas.io.json import json_normalize 
import matplotlib.cm as cm
import matplotlib.colors as colors
# import k-means from clustering stage
from sklearn.cluster import KMeans
import requests # library to handle requests
import lxml.html as lh
import bs4 as bs
import urllib.request
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


sg_df = pd.read_csv('/Users/shiqinchoo/Desktop/resources/sg_grouped.csv')
kl_df = pd.read_csv('/Users/shiqinchoo/Desktop/resources/kl_grouped.csv')


# In[5]:


sg_df.head()


# In[6]:


kl_df.head()


# In[7]:


sg_df = sg_df.dropna()


# In[9]:


sg_df.head()


# In[10]:


kl_df = kl_df.dropna()


# In[11]:


kl_df.head()


# In[13]:


sg_df = sg_df.drop(['Unnamed: 0'], axis = 1)


# In[14]:


sg_df.head()


# In[15]:


kl_df = kl_df.drop(['Unnamed: 0'], axis = 1)
kl_df.head()


# In[23]:


sg_list = sg_df.columns
len(sg_list)


# In[26]:


sg_list = sg_list.drop(['Neighborhood'])
sg_list


# In[28]:


kl_list = kl_df.columns
kl_list = kl_list.drop(['Neighborhood'])
kl_list


# In[50]:


from sklearn.preprocessing import MinMaxScaler
sg_x = sg_df[sg_list].values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_sgx = min_max_scaler.fit_transform(sg_x)
feature_sgx [0:5]


# In[51]:


sg_dist_matrix = distance_matrix(feature_sgx,feature_sgx) 
print(dist_matrix)


# In[43]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[ ]:





# In[59]:


import pylab
fig = pylab.figure(figsize=(30,20))
def llf(id):
    return '[%s]' % (sg_df['Neighborhood'][id] )
    
dendro = hierarchy.dendrogram(sg_Z,  leaf_label_func=llf, leaf_rotation=45, leaf_font_size =12, orientation = 'top')


# In[52]:




sg_Z = hierarchy.linkage(sg_dist_matrix, 'complete')
sg_dendro = dendrogram(sg_Z)


# In[53]:


sg_Z = hierarchy.linkage(sg_dist_matrix, 'average')
dendro = hierarchy.dendrogram(sg_Z)


# In[33]:


kl_x = kl_df[kl_list].values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_klx = min_max_scaler.fit_transform(kl_x)
feature_klx [0:5]


# In[35]:


dist_matrix = distance_matrix(feature_klx,feature_klx) 
print(dist_matrix)


# In[36]:


kl_dist_matrix = distance_matrix(feature_klx,feature_klx) 
print(kl_dist_matrix)


# In[37]:


kl_Z = hierarchy.linkage(kl_dist_matrix, 'complete')
dendro = hierarchy.dendrogram(kl_Z)


# In[153]:


import pylab
kl_fig = pylab.figure(figsize=(30,20))
def kl_llf(id):
    return '[%s]' % (kl_df['Neighborhood'][id] )
    
dendro = hierarchy.dendrogram(kl_Z,  leaf_label_func=kl_llf, leaf_rotation=45, leaf_font_size =12, orientation = 'top')


# In[62]:


import pandas as pd
sg_n = pd.read_csv('/Users/shiqinchoo/Desktop/resources/sg_towns.csv')
kl_n = pd.read_csv('/Users/shiqinchoo/Desktop/resources/kl_districts.csv')


# In[66]:


sg_n


# In[68]:


sg_new = pd.DataFrame()
sg_new['Name'] = sg_n['Neighbourhoods']
sg_new['Latitutde'] = sg_n['Latitude']
sg_new['Longitude'] = sg_n['Longitude']
sg_new


# In[67]:


kl_n


# In[69]:


kl_new = pd.DataFrame()
kl_new['Name'] = kl_n['Name']
kl_new['Latitude'] = kl_n['Latitudes']
kl_new['Longitude'] = kl_n['Longitudes']
kl_new


# In[85]:


list(sg_new['Name'])


# In[86]:


list(kl_new['Name'])


# In[89]:


sg_name_list = list(sg_new['Name'])
kl_name_list = list(kl_new['Name'])
cities_list = sg_name_list + kl_name_list
cities_list


# In[90]:


len(cities_list)


# In[92]:


sg_name_lat = list(sg_new['Latitutde'])
kl_name_lat = list(kl_new['Latitude'])
cities_lat = sg_name_lat + kl_name_lat
sg_name_long = list(sg_new['Longitude'])
kl_name_long = list(kl_new['Longitude'])
cities_long = sg_name_long + kl_name_long


# In[93]:


cities = pd.DataFrame()
cities['Areas'] = cities_list
cities['Latitude'] = cities_lat
cities['Longitude'] = cities_long
cities


# In[94]:


#creating the API call
CLIENT_ID = 'FMOVI2TV0X0ADLHYSKK2NDDWF2AL5PZJ2DWEII2G11OSFH1S' # your Foursquare ID
CLIENT_SECRET = '0IG2HMFEHHN4LO1JOBN1SIB4LGA24HDGOGGMA3JVMPCIUICV' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version


# In[95]:


#creating a helper function for the api call to obtain the venues around the neighbourhoods
LIMIT = 50
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[97]:


#calling the API for the variou neighbourhoods. 
cities_venues = getNearbyVenues(names=cities['Areas'],
                                   latitudes=cities['Latitude'],
                                   longitudes=cities['Longitude']
                                  )


# In[106]:


print('The shape of venues is: ', cities_venues.shape)
cities_venues.head(20)


# In[107]:


# one hot encoding - vectorizing the data
venues_onehot = pd.get_dummies(cities_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
venues_onehot['Neighborhood'] = cities_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [venues_onehot.columns[-1]] + list(venues_onehot.columns[:-1])
venues_onehot = venues_onehot[fixed_columns]

venues_onehot.head()


# In[108]:


venues_grouped = venues_onehot.groupby('Neighborhood').mean().reset_index()
venues_grouped


# In[165]:


venues_grouped.to_csv('combined_cities_df.csv')


# In[122]:


is_punggol = venues_grouped['Neighborhood']=='Punggol'
punggol = venues_grouped[is_punggol]
punggol


# In[123]:


punggol = punggol.drop(columns=['Neighborhood'],axis = 1)
punggol


# In[129]:


data_grouped = venues_grouped.drop(columns=['Neighborhood'], axis = 1)
data_grouped


# In[166]:


var = data_grouped.columns
var


# In[160]:


venues_group = venues_grouped[var].values #returns a numpy array


# In[161]:


min_max_scaler = MinMaxScaler()
features_cities = min_max_scaler.fit_transform(venues_group)
features_cities[0:5]


# In[162]:


features_matrix = distance_matrix(features_cities,features_cities) 
print(features_matrix)


# In[163]:


cities_Z = hierarchy.linkage(features_matrix, 'complete')


# In[164]:



cities_fig = pylab.figure(figsize=(30,20))
def cities_llf(id):
    return '[%s]' % (venues_grouped['Neighborhood'][id] )
    
cities_dendro = hierarchy.dendrogram(cities_Z,  leaf_label_func=cities_llf, leaf_rotation=45, leaf_font_size =12, orientation = 'top')


# In[130]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:





# In[133]:


corr = cosine_similarity(data_grouped)


# In[135]:


names = [i for i in list(cities['Areas'])]


# In[136]:


len(names)


# In[140]:


df = pd.DataFrame(corr, columns = names, index=names)
df.head()


# In[141]:


df


# In[142]:


df.to_csv('towns_similarity.csv')


# In[148]:


amk_sim = df['Ang Mo Kio'].iloc[27:].to_frame()
amk_sim.head()


# In[149]:


amk_sim.sort_values(['Ang Mo Kio'],ascending=False)


# In[150]:


klang_sim = df['Port Klang'].iloc[:27].to_frame().sort_values(['Port Klang'], ascending=False)


# In[151]:


klang_sim


# In[ ]:




