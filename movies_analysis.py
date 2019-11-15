#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import itertools
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy as sp
from scipy import stats
from scipy.stats import sem, t
from statsmodels.stats import weightstats as stests

df = pd.read_csv('movies.csv')


# In[34]:


#keeping only the necessary columns
df = df[['Title', 'Worldwide Gross', 'Production Budget', 'Release Date', 'Major Genre', 'Director', 'Rotten Tomatoes Rating', 'IMDB Rating', 'IMDB Votes']]
#renaming some columns
df.rename(columns = {'Worldwide Gross' : 'Gross', 'Production Budget' : 'Budget', 'Release Date' : 'Date', 'Major Genre' : 'Genre', 'Rotten Tomatoes Rating' : 'RTRating', 'IMDB Rating' : 'IMDBRating', 'IMDB Votes' : 'IMDBVotes'}, inplace = True)


# In[35]:


#cropping unnecessary info from Date
df.Date = df['Date'].str.rstrip()
df.Date = df['Date'].str[-2:]


# In[36]:


#pruning unwanted rows from Date
df = df[df['Date'].str.isdecimal() == True]

#pruning unwanted rows from Gross
df = df[df['Gross'] != 'Unknown']


# In[37]:


#fixing indices after pruning (starting from 1 instead of 0)
df.index = np.arange(1, len(df) + 1)


# In[38]:


#fixing Date format
df.Date = pd.to_numeric(df['Date'], errors = 'coerce')
df.Date = df['Date'].map("{:02}".format)
df.Date = df['Date'].apply(lambda x:'20'+x if 0 <= int(x) <= 19 else '19'+x)
df.Date = df['Date'].astype(int)


# In[39]:


#converting Gross from str to float
df.Gross = df['Gross'].astype(float)


# In[40]:


#fixing scale climax on RTRating
df.RTRating = df['RTRating'].apply(lambda x: x/10)


# In[41]:


#making a genres Dataframe splitting the one column to two (Columns: ID, First, Second)
#helpful 
genres_df = df['Genre'].dropna()
genres_df = genres_df.str.split('/', expand = True)
genres_df.columns = ['First', 'Second']

second_genre_df = genres_df['Second'].dropna()
print(genres_df)


# In[42]:


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


# In[43]:


#making the Genre Number Dataframe so as to make the bar plot later
genre_numbers_df = pd.DataFrame.from_dict(genres_hash, orient = 'index')
genre_numbers_df.columns = ['Number of Movies']
genre_numbers_df.insert(0, 'Genre', genres_hash.keys())
genre_numbers_df.sort_values('Number of Movies', ascending = False, inplace = True)
genre_numbers_df.reset_index()
genre_numbers_df.index = range(len(genre_numbers_df))
print(genre_numbers_df)


# In[44]:


#to csv
df.to_csv('cleandata_movies.csv', index_label = 'ID')
print(df)


# In[45]:


#making Worldwide Gross histogram
sns.set_context('paper')
sns.set_style('white')

gross = df['Gross']

sns.distplot( gross, kde = False, color = 'darkgreen', bins = 100)
plt.xscale('log')
plt.title('Worldwide Gross', color = 'darkgreen', fontsize = 18)
plt.xlabel('Dollars', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('WorldwideGross_Histogram.png')


# In[46]:


#making Rotten Tomatoes Rating histogram
rtr = df['RTRating'].dropna()

sns.distplot(rtr, kde = False, color='darkred', bins = 10)
plt.title('Rotten Tomatoes Rating',color = 'darkred', fontsize = 18)
plt.xlabel('Rating', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

mean = rtr.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('RottenTomatoesRating_Histogram.png')


# In[47]:


#making IMDB Rating histogram
imdbr = df['IMDBRating'].dropna()

sns.distplot(imdbr, kde = False, color='goldenrod', bins = 8)

plt.xlim(0, 10)
plt.title('IMDB Rating',color = 'goldenrod', fontsize = 18)
plt.xlabel('Rating', fontsize = 14)
plt.ylabel('Number of Movies', fontsize=14)

mean = imdbr.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('IMDBRating_Histogram.png')


# In[48]:


#making IMDB Votes histogram
imdbv = df['IMDBVotes'].dropna()

sns.distplot(imdbv, kde = False, color = 'black', bins = 25)
plt.title('IMDB Votes',color = 'black', fontsize = 18)
plt.xlabel('Number of Votes', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

mean = imdbv.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('IMDBVotes_Histogram.png')


# In[49]:


#making Genres bar plot
sns.barplot(x = 'Number of Movies', y = 'Genre', data = genre_numbers_df)
plt.title('Major Genre', color = 'black', fontsize = 18)

plt.savefig('num_movie_genre_barplot.png')


# In[50]:


#merging Gross with IMDBVotes into one DataFrame
gross_votes_df = pd.merge(pd.DataFrame(gross), pd.DataFrame(imdbv), left_index = True, right_index = True)
print(gross_votes_df)


# In[53]:


m = gross_votes_df['Gross'].mean()

sns.distplot(gross_votes_df['Gross'], kde = False, color = 'darkgreen', label = 'Gross', bins = 90)

plt.xscale('log')
plt.yscale('log')
plt.xlim(m)
#.mean(skipna = True)

plt.title('Worldwide Gross',color = 'black', fontsize = 18)
plt.xlabel('Dollars [0-mean()] ', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('Grosslog_Histogram.png')


# In[20]:


m = gross_votes_df['IMDBVotes'].mean()
sns.distplot(gross_votes_df['IMDBVotes'] , kde = False, color = 'black', label = 'IMDBVotes', bins = 80)

plt.xscale('log')
plt.yscale('log')
plt.xlim(m)

plt.title('IMDB Votes',color = 'black', fontsize = 18)
plt.xlabel('Votes [18-mean()] ', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('Voteslog_Histogram.png')


# In[21]:


#making Worldwide Gross and IMDB Votes Scatterplot
sns.scatterplot(x = 'IMDBVotes', y = 'Gross', data = gross_votes_df, facecolor = 'goldenrod' )

plt.xscale('log')
plt.title('IMDB Votes - Worldwide Gross',color = 'black', fontsize = 18)
plt.xlabel('Votes', fontsize = 14)
plt.ylabel('Gross ($)', fontsize = 14)

plt.savefig('VotesGross_Scatterplot.png')


# In[22]:


#Pearson Correlation Coefficient: Worldwide Gross and IMDB Votes
gross_votes_df.corr(method = 'pearson')


# In[23]:


#Spearman Correlation Coefficient: Worldwide Gross and IMDB Votes
gross_votes_df.corr(method = 'spearman')


# In[54]:


#2 sample z-test: Worldwide Gross and IMDB Votes
print('H0: A popular movie at the cinema, is also popular online.\n ')

ztest, pval = stests.ztest(gross_votes_df['Gross'], x2 = gross_votes_df['IMDBVotes'], value = gross_votes_df['Gross'].mean(), alternative = 'two-sided')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H0)')

else:
    
    print('\nAccept Null Hypothesis (H0)')


# In[55]:


#concatenating RTRating with IMDBRating into one DataFrame
rtr_imdbr_df = pd.concat([pd.DataFrame(rtr), pd.DataFrame(imdbr)], axis =1)
rtr_imdbr_df.dropna(inplace=True)
print(rtr_imdbr_df)


# In[26]:


#making RTRating and IMDBRating Scatterplot
sns.scatterplot(x = 'RTRating', y='IMDBRating', data = rtr_imdbr_df, facecolor = 'm', edgecolor = 'black')

plt.title('Rotten Tomatoes Rating - IMDB Rating',color = 'black', fontsize = 18)
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel('Rotten Tomatoes Rating', fontsize = 14)
plt.ylabel('IMDB Rating', fontsize = 14)

plt.savefig('RTR_IMDB_Ratings_Scatterplot.png')


# In[27]:


#Pearson Correlation Coefficient: RTRating and IMDBRating
rtr_imdbr_df.corr(method = 'pearson')


# In[28]:


#Spearman Correlation Coefficient: RTRating and IMDBRating
rtr_imdbr_df.corr(method = 'spearman')


# In[56]:


#2 sample z-test:  RTRating and IMDBRating
print('H0: The way people vote in Rotten Tomatoes and IMDB has a non-linear correlation.\n ')

ztest, pval = stests.ztest(rtr_imdbr_df['RTRating'], x2 = rtr_imdbr_df['IMDBRating'], value = rtr_imdbr_df['RTRating'].mean() , alternative = 'two-sided')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H0)')

else:
    
    print('\nAccept Null Hypothesis (H0)')


# In[30]:


#merging Gross with Genres into one DataFrame
gross_genre_df = df[['Gross']]
gross_genre_df['Genre1'] = genres_df['First']
gross_genre_df.dropna(inplace = True)
gross_genre_df['Genre2'] = genres_df['Second']
gross_genre_df = gross_genre_df.replace(np.nan, ' ', regex=True)
print(gross_genre_df)


# In[32]:


#creating a dictionary for Genre: total Gross per genre

gross_hash ={}

gross_list = gross_genre_df['Gross'].values
genre1_list = gross_genre_df['Genre1'].values
genre2_list = gross_genre_df['Genre2'].values

for g in range(len(gross_list)):

    
    if genre1_list[g] != ' ':
    
        if genre1_list[g] not in gross_hash:
            
            
            gross_hash[genre1_list[g]] = [gross_list[g], 1]

           

        else:
            
            gross_sum = float(list(gross_hash.get(genre1_list[g]))[0]) + gross_list[g]
            counter_movies = int(list(gross_hash.get(genre1_list[g]))[1]) + 1
            
            gross_hash[genre1_list[g]] = [gross_sum, counter_movies]

            
            
    if genre2_list[g] != ' ':
        
        if genre2_list[g] not in gross_hash:

            gross_hash[genre2_list[g]] = [gross_list[g], 1 ]
          

        else:
            
            gross_sum = float(list(gross_hash.get(genre2_list[g]))[0]) + gross_list[g]
            counter_movies = int(list(gross_hash.get(genre2_list[g]))[1]) + 1
                                  
            gross_hash[genre2_list[g]] = [gross_sum, counter_movies]




print(gross_hash)


# In[ ]:





# In[ ]:




