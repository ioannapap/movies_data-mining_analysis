#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import itertools
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy as sp
from scipy import stats

df = pd.read_csv('movies.csv')


# In[3]:


#removing unnecessary columns
df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'US Gross', 'US DVD Sales', 'MPAA Rating', 'Running Time (min)', 'Distributor', 'Source', 'Creative Type'], axis = 1, inplace = True)

#renaming some columns
df.rename(columns = {'Worldwide Gross' : 'Gross', 'Production Budget' : 'Budget', 'Release Date' : 'Date', 'Major Genre' : 'Genre', 'Rotten Tomatoes Rating' : 'RTRating', 'IMDB Rating' : 'IMDBRating', 'IMDB Votes' : 'IMDBVotes'}, inplace = True)


# In[4]:


#cropping unnecessary info from Date
df.Date = df['Date'].str.rstrip()
df.Date = df['Date'].str[-2:]


# In[5]:


#pruning unwanted rows from Date
df = df[df['Date'].str.isdecimal() == True]

#pruning unwanted rows from Gross
df = df[df['Gross'] != 'Unknown']


# In[6]:


#fixing indices after pruning (starting from 1 instead of 0)
df.index = np.arange(1, len(df) + 1)


# In[7]:


#fixing Date format
df.Date = pd.to_numeric(df['Date'], errors = 'coerce')
df.Date = df['Date'].map("{:02}".format)
df.Date = df['Date'].apply(lambda x:'20'+x if 0 <= int(x) <= 19 else '19'+x)
df.Date = df['Date'].astype(int)


# In[8]:


#converting Gross from str to float
df.Gross = df['Gross'].astype(float)


# In[9]:


#fixing scale climax on RTRating
df.RTRating = df['RTRating'].apply(lambda x: x/10)


# In[10]:


#making a genres Dataframe splitting the one column to two (Columns: ID, First, Second)
#helpful 
genres_df = df['Genre'].dropna()
genres_df = genres_df.str.split('/', expand = True)
genres_df.columns = ['First', 'Second']

second_genre_df = genres_df['Second'].dropna()


# In[11]:


#constructing a dictionary for Genre (key = genre : value = number of movies)
genres_hash = {}

for i in genres_df['First']:

    if i not in genres_hash:

        genres_hash[i] = 1

    else:

        genres_hash[i] += 1


for i in second_genre_df:

    if i not in genres_hash:

        genres_hash[i] = 1

    else:

        genres_hash[i] += 1


# In[12]:


#making the Genre Number Dataframe so as to make the bar plot later
genre_numbers_df = pd.DataFrame.from_dict(genres_hash, orient = 'index')
genre_numbers_df.columns = ['Number of Movies']
genre_numbers_df.insert(0, 'Genre', genres_hash.keys())
genre_numbers_df.sort_values('Number of Movies', ascending = False, inplace = True)
genre_numbers_df.reset_index()
genre_numbers_df.index = range(len(genre_numbers_df))
print(genre_numbers_df)


# In[13]:


#to csv
df.to_csv('cleandata_movies.csv', index_label = 'ID')
print(df)


# In[59]:


#making Worldwide Gross histogram
sns.set_context('paper')
sns.set_style('white')

gross_df = df['Gross']

sns.distplot( gross_df, kde = False, color = 'darkgreen', bins = 100)
plt.xscale('log')
plt.title('Worldwide Gross', color = 'darkgreen', fontsize = 18)
plt.xlabel('Dollars', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('WorldwideGross_Histogram.png')


# In[15]:


#making Rotten Tomatoes Rating histogram
rtr_df = df['RTRating'].dropna()

sns.distplot(rtr_df, kde = False, color='darkred', bins = 10)
plt.title('Rotten Tomatoes Rating',color = 'darkred', fontsize = 18)
plt.xlabel('Rating', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

mean = rtr_df.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('RottenTomatoesRating_Histogram.png')


# In[16]:


#making IMDB Rating histogram
imdbr_df = df['IMDBRating'].dropna()

sns.distplot(imdbr_df, kde = False, color='goldenrod', bins = 8)

plt.xlim(0, 10)
plt.title('IMDB Rating',color = 'goldenrod', fontsize = 18)
plt.xlabel('Rating', fontsize = 14)
plt.ylabel('Number of Movies', fontsize=14)

mean = imdbr_df.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('IMDBRating_Histogram.png')


# In[17]:


#making IMDB Votes histogram
imdbv_df = df['IMDBVotes'].dropna()

sns.distplot(imdbv_df, kde = False, color = 'black', bins = 25)
plt.title('IMDB Votes',color = 'black', fontsize = 18)
plt.xlabel('Number of Votes', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

mean = imdbv_df.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('IMDBVotes_Histogram.png')


# In[18]:


#making Genres bar plot
sns.barplot(x = 'Number of Movies', y = 'Genre', data = genre_numbers_df)
plt.title('Major Genre', color = 'black', fontsize = 18)

plt.savefig('num_movie_genre_barplot.png')


# In[19]:


#merging Gross with Votes 
gross_votes_df = pd.merge(pd.DataFrame(gross_df), pd.DataFrame(imdbv_df), left_index = True, right_index = True)
#gross_votes_df = gross_votes_df.dropna()
print(gross_votes_df)


# In[56]:


#NEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEDS EDIT
sns.distplot(gross_votes_df['Gross'], kde = False, color = 'darkgreen', label = 'Gross', bins = 100)

plt.xscale('log')
plt.yscale('log')
#.mean(skipna = True)

plt.title('Worldwide Gross',color = 'black', fontsize = 18)
plt.xlabel('Dollars (mean number) ', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('Grosslog_Histogram.png')


# In[65]:


#NEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEDS EDIT
sns.distplot(gross_votes_df['IMDBVotes'] , kde = False, color = 'black', label = 'IMDBVotes', bins = 20)

plt.xscale('log')
plt.yscale('log')

plt.title('IMDB Votes',color = 'black', fontsize = 18)
plt.xlabel('Votes (mean number) ', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('Voteslog_Histogram.png')


# In[104]:


#making Worldwide Gross and IMDB Votes Scatterplot

sns.scatterplot(x = 'IMDBVotes', y = 'Gross', data = gross_votes_df, facecolor = 'goldenrod' )

plt.xscale('log')
plt.title('IMDB Votes - Worldwide Gross',color = 'black', fontsize = 18)

plt.xlabel('Votes', fontsize = 14)
plt.ylabel('Gross', fontsize = 14)

plt.savefig('VotesGross_Scatterplot.png')


# In[ ]:




