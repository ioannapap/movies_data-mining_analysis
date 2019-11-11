#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#import scipy as sp
#from scipy import stats
df = pd.read_csv('movies.csv')


# In[2]:


#removing unnecessary columns
df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'US Gross', 'US DVD Sales', 'MPAA Rating', 'Running Time (min)', 'Distributor', 'Source', 'Creative Type'], axis = 1, inplace = True)

#renaming some columns
df.rename(columns = {'Worldwide Gross' : 'Gross', 'Production Budget' : 'Budget', 'Release Date' : 'Date', 'Major Genre' : 'Genre', 'Rotten Tomatoes Rating' : 'RTRating', 'IMDB Rating' : 'IMDBRating', 'IMDB Votes' : 'IMDBVotes'}, inplace = True)


# In[3]:


#cropping unnecessary info from Date
df.Date = df['Date'].str.rstrip()
df.Date = df['Date'].str[-2:]


# In[4]:


#pruning unwanted rows from Date
df = df[df['Date'].str.isdecimal() == True]

#pruning unwanted rows from Gross
df = df[df['Gross'] != 'Unknown']


# In[5]:


#fixing indices after pruning (starting from 1 instead of 0)
df.index = np.arange(1, len(df) + 1)


# In[6]:


#fixing Date format
df.Date = pd.to_numeric(df['Date'], errors = 'coerce')
df.Date = df['Date'].map("{:02}".format)
df.Date = df['Date'].apply(lambda x:'20'+x if 0 <= int(x) <= 19 else '19'+x)
df.Date = df['Date'].astype(int)


# In[7]:


#converting Gross from str to float
df.Gross = df['Gross'].astype(float)


# In[8]:


#fixing scale climax on RTRating
df.RTRating = df['RTRating'].apply(lambda x: x/10)


# In[9]:


#to csv
df.to_csv('cleandata_movies.csv', index_label = 'ID')
print(df)


# In[10]:


sns.set_context('paper')
sns.set_style('white')

sns.distplot(df['Gross']/1000000, kde=False, color='darkgreen', bins=80)
plt.ylim(0,1700)
plt.xlim(0,600)
plt.title('Worldwide Gross', color='darkgreen', fontsize=18)
plt.xlabel('Millions', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.savefig('WorldwideGross_Histogram.png')


# In[11]:


rtr_df = df['RTRating'].dropna()

sns.distplot(rtr_df, kde=False, color='darkred', bins=10)
plt.title('Rotten Tomatoes Rating',color='darkred', fontsize=18)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)

plt.savefig('RottenTomatoesRating_Histogram.png')


# In[12]:


imdbr_df = df['IMDBRating'].dropna()

sns.distplot(imdbr_df, kde=False, color='goldenrod', bins=8)
plt.xlim(0, 10)
plt.title('IMDB Rating',color='goldenrod', fontsize=18)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)

plt.savefig('IMDBRating_Histogram.png')


# In[13]:


imdbv_df = df['IMDBVotes'].dropna()

sns.distplot(imdbv_df, kde=False, color='black', bins=25)
plt.title('IMDB Votes',color='black', fontsize=18)
plt.xlabel('Number of Votes', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)

plt.savefig('IMDBVotes_Histogram.png')


# In[14]:


genres_df = df['Genre'].dropna()

genres_df = genres_df.str.split('/', expand = True)
genres_df.columns = ['First', 'Second']


print(genres_df)

#to csv
genres_df.to_csv('movie_genres.csv', index_label = 'ID')



second_genre_df = genres_df['Second'].dropna()

#constructing dictionary for movie Genres
genres_hash = {}

for i in genres_df['First']:
    
    if i not in genres_hash:
        
        genres_hash[i] = 1
    
    else:
        
        genres_hash[i] += 1
        
print(genres_hash)
print(len(genres_hash))

for i in second_genre_df:

    if i not in genres_hash:
        
        genres_hash[i] = 1
    
    else:
        
        genres_hash[i] += 1
    
print(genres_hash)
print(len(genres_hash))


# In[ ]:




