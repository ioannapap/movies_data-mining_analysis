#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy as sp
from scipy import stats
from scipy.stats import sem, t
from pandas import Series, DataFrame
from statsmodels.stats import weightstats as stests

pd.options.display.float_format = '{:.1f}'.format
df = pd.read_csv('movies.csv')


# In[3]:


#keeping only the necessary columns
df = df[['Title', 'Worldwide Gross', 'Production Budget', 'Release Date', 'Major Genre', 'Director', 'Rotten Tomatoes Rating', 'IMDB Rating', 'IMDB Votes']]
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


#to csv
df.to_csv('cleandata_movies.csv', index_label = 'ID')
print(df)


# In[11]:


#making Production Budget and GrossDataframe ---my data mining problem ---
budget_gross_df = df[['Gross', 'Budget']]
budget_gross_df = budget_gross_df.dropna()
print(budget_gross_df)


# In[12]:


#making a genres Dataframe splitting the one column to two (Columns: ID, First, Second)
#helpful 
genres_df = df['Genre'].dropna()
genres_df = genres_df.str.split('/', expand = True)
genres_df.columns = ['First', 'Second']

second_genre = genres_df['Second'].dropna()

print(genres_df)


# In[13]:


#constructing a dictionary for Genre (key = genre : value = number of movies)
genres_hash = {}

for i in genres_df['First']:

    if i not in genres_hash:

        genres_hash[i] = 1

    else:

        genres_hash[i] += 1


for i in second_genre:

    if i not in genres_hash:

        genres_hash[i] = 1

    else:

        genres_hash[i] += 1


# In[14]:


#making the Genre Number Dataframe so as to make the bar plot later
genre_numbers_df = pd.DataFrame.from_dict(genres_hash, orient = 'index')
genre_numbers_df.columns = ['Number of Movies']
genre_numbers_df.sort_values('Number of Movies', ascending = False, inplace = True)
genre_numbers_df.reset_index(level = 0, inplace = True)
genre_numbers_df = genre_numbers_df.rename(columns = {'index' : 'Genre'})
print(genre_numbers_df)


# In[15]:


#making Worldwide Gross histogram
sns.set_context('paper')
sns.set_style('white')

gross = df['Gross']

plt.figure(figsize=(10, 5))
sns.distplot( gross, kde = False, color = 'darkgreen', bins = 30)

plt.title('Worldwide Gross', color = 'darkgreen', fontsize = 18)
plt.xlabel('Dollars', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('WorldwideGross_Histogram.png')


# In[16]:


#making Rotten Tomatoes Rating histogram
rtr = df['RTRating'].dropna()

plt.figure(figsize=(10, 5))
sns.distplot(rtr, kde = False, color='darkred', bins = 10)

plt.title('Rotten Tomatoes Rating',color = 'darkred', fontsize = 18)
plt.xlabel('Rating', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

mean = rtr.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('RottenTomatoesRating_Histogram.png')


# In[17]:


#making IMDB Rating histogram
imdbr = df['IMDBRating'].dropna()

plt.figure(figsize=(10, 5))
sns.distplot(imdbr, kde = False, color='goldenrod', bins = 8)

plt.xlim(0, 10)
plt.title('IMDB Rating',color = 'goldenrod', fontsize = 18)
plt.xlabel('Rating', fontsize = 14)
plt.ylabel('Number of Movies', fontsize=14)

mean = imdbr.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('IMDBRating_Histogram.png')


# In[18]:


#making IMDB Votes histogram
imdbv = df['IMDBVotes'].dropna()

plt.figure(figsize=(10, 5))
sns.distplot(imdbv, kde = False, color = 'black', bins = 25)

plt.title('IMDB Votes',color = 'black', fontsize = 18)
plt.xlabel('Number of Votes', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

mean = imdbv.mean()
plt.axvline(mean, color='r', linestyle='--')
plt.legend({'Mean':mean})

plt.savefig('IMDBVotes_Histogram.png')


# In[19]:


#making Genres bar plot
plt.figure(figsize=(10, 5))

sns.barplot(x = 'Number of Movies', y = 'Genre', saturation = 1, data = genre_numbers_df)

plt.title('Major Genre', color = 'black', fontsize = 18)

plt.savefig('num_movie_genre_barplot.png')


# In[20]:


#merging Gross with IMDBVotes into one DataFrame
gross_votes_df = pd.merge(pd.DataFrame(gross), pd.DataFrame(imdbv), left_index = True, right_index = True)
print(gross_votes_df)
print(gross_votes_df['Gross'].mean())


# In[21]:


#making exp. Worldwide Gross Histogram
plt.figure(figsize=(10, 5))
sns.distplot(gross_votes_df['Gross'], kde = False, color = 'darkgreen', label = 'Gross', bins = 50)

plt.xscale('log')
plt.yscale('log')

plt.title('Worldwide Gross exp.',color = 'black', fontsize = 18)
plt.xlabel('Dollars ', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('Grosslog_Histogram.png')


# In[22]:


#making exp. IMDB Votes Histogram
plt.figure(figsize=(10, 5))
sns.distplot(gross_votes_df['IMDBVotes'] , kde = False, color = 'black', label = 'IMDBVotes', bins = 50)

plt.xscale('log')
plt.yscale('log')

plt.title('IMDB Votes exp.',color = 'black', fontsize = 18)
plt.xlabel('Votes ', fontsize = 14)
plt.ylabel('Number of Movies', fontsize = 14)

plt.savefig('Voteslog_Histogram.png')


# In[23]:


#making exp. Worldwide Gross and IMDB Votes Scatterplot

plt.figure(figsize=(10, 5))
sns.scatterplot(x = 'IMDBVotes', y = 'Gross', data = gross_votes_df, facecolor = 'goldenrod' )

plt.xscale('log')
plt.yscale('log')
plt.ylim(20)

plt.title('IMDB Votes - Worldwide Gross',color = 'black', fontsize = 18)
plt.xlabel('Votes', fontsize = 14)
plt.ylabel('Gross ($)', fontsize = 14)

plt.savefig('VotesGrosslog_Scatterplot.png')


# In[24]:


#making Worldwide Gross and IMDB Votes Scatterplot

plt.figure(figsize=(10, 5))
sns.scatterplot(x = 'IMDBVotes', y = 'Gross', data = gross_votes_df, facecolor = 'goldenrod' )

plt.title('IMDB Votes - Worldwide Gross',color = 'black', fontsize = 18)
plt.xlabel('Votes', fontsize = 14)
plt.ylabel('Gross ($)', fontsize = 14)

plt.savefig('VotesGross_Scatterplot.png')


# In[25]:


#Pearson Correlation Coefficient: Worldwide Gross and IMDB Votes
pd.options.display.float_format = '{:.6f}'.format
gross_votes_df.corr(method = 'pearson')


# In[26]:


#Spearman Correlation Coefficient: Worldwide Gross and IMDB Votes
gross_votes_df.corr(method = 'spearman')


# In[27]:


#2 sample z-test: Worldwide Gross and IMDB Votes
print('H0: Movies are more popular online, than in the cinema.\n ')

ztest, pval = stests.ztest(gross_votes_df['IMDBVotes'], x2 = gross_votes_df['Gross'], value = 0, alternative = 'larger')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H0)')

else:
    
    print('\nAccept Null Hypothesis (H0)')
    
pd.options.display.float_format = '{:.1f}'.format


# In[28]:


#concatenating RTRating with IMDBRating into one DataFrame
rtr_imdbr_df = pd.concat([pd.DataFrame(rtr), pd.DataFrame(imdbr)], axis =1)
rtr_imdbr_df.dropna(inplace=True)
print(rtr_imdbr_df)


# In[29]:


#making RTRating and IMDBRating Scatterplot
plt.figure(figsize=(10, 5))
sns.scatterplot(x = 'RTRating', y='IMDBRating', data = rtr_imdbr_df, facecolor = 'm', edgecolor = 'black')

plt.title('Rotten Tomatoes Rating - IMDB Rating',color = 'black', fontsize = 18)
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel('Rotten Tomatoes Rating', fontsize = 14)
plt.ylabel('IMDB Rating', fontsize = 14)

plt.savefig('RTR_IMDB_Ratings_Scatterplot.png')


# In[30]:


#Pearson Correlation Coefficient: RTRating and IMDBRating
pd.options.display.float_format = '{:.6f}'.format
rtr_imdbr_df.corr(method = 'pearson')


# In[31]:


#Spearman Correlation Coefficient: RTRating and IMDBRating
rtr_imdbr_df.corr(method = 'spearman')


# In[32]:


#2 sample z-test1:  RTRating and IMDBRating
print('H0: People in Rotten Tomatoes and  IMDB vote similarly.\n ')

ztest, pval = stests.ztest(rtr_imdbr_df['RTRating'], x2 = rtr_imdbr_df['IMDBRating'], value = 0 , alternative = 'two-sided')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H0)')

else:
    
    print('\nAccept Null Hypothesis (H0)')


# In[33]:


#2 sample z-test2:  RTRating and IMDBRating
print('H1: People in Rotten Tomatoes vote higher than in IMDB.\n ')

ztest, pval = stests.ztest(rtr_imdbr_df['RTRating'], x2 = rtr_imdbr_df['IMDBRating'], value = 0 , alternative = 'larger')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H1)')

else:
    
    print('\nAccept Null Hypothesis (H1)')
    
pd.options.display.float_format = '{:.1f}'.format


# In[34]:


#making one DataFrame with Genres (1 and 2 ) and Gross
gross_genre_df = pd.DataFrame(df['Gross'])
gross_genre_df['Genre1'] = genres_df[['First']]
gross_genre_df.dropna(inplace = True)
gross_genre_df['Genre2'] = genres_df[['Second']]


# In[35]:


#Grouping by genres and finding mean gross (in both genre dfs), then putting them all in one DataFrame
m_gross_genre1_df = gross_genre_df.groupby('Genre1').mean().reset_index()
m_gross_genre2_df = gross_genre_df.groupby('Genre2').mean().reset_index()
m_gross_genre2_df = m_gross_genre2_df.rename(columns = {'Genre2' : 'Genre1'})
m_gross_genre1_df = m_gross_genre1_df.append(m_gross_genre2_df, ignore_index = True)
m_gross_genre1_df = m_gross_genre1_df.rename(columns = {'Genre1' : 'Genre', 'Gross' : 'Mean Gross'})

m_gross_genre1_df.sort_values('Mean Gross', ascending = False, inplace = True)


# In[36]:


#Grouping by genres and finding std gross (in both genre dfs), then putting them all in one DataFrame
std_gross_genre1_df = gross_genre_df.groupby('Genre1').std().reset_index()
std_gross_genre2_df = gross_genre_df.groupby('Genre2').std().reset_index()
std_gross_genre2_df = std_gross_genre2_df.rename(columns = {'Genre2' : 'Genre1'})

std_gross_genre1_df = std_gross_genre1_df.append(std_gross_genre2_df, ignore_index = True)
std_gross_genre1_df = std_gross_genre1_df.rename(columns = {'Genre1' : 'Genre', 'Gross' : 'Std Gross'})


# In[37]:


#calculating the confidence intervals ci lower and ci upper 
conf_int = stats.norm.interval(0.95, loc = m_gross_genre1_df['Mean Gross'], scale = std_gross_genre1_df['Std Gross']/np.sqrt(std_gross_genre1_df['Std Gross'].count()))
conf_int = np.array(conf_int)
print(conf_int)


# In[38]:


# putting ci lower and ci upper in or m_gross_genre1_df DataFrame
m_gross_genre1_df['Bottom Error Ci'] = conf_int[0]
m_gross_genre1_df['Top Error Ci'] = conf_int[1]
m_gross_genre1_df.reset_index(drop = True, inplace = True)
print(m_gross_genre1_df)


# In[39]:


# fixing the negative bottom error CIs and give them the Mean Gross number
# in order to be 0 when the errorbar calculates xerror ( it calculates: mean - bottom error ci)
m_gross_genre1_df['Bottom Error Ci'] = m_gross_genre1_df['Bottom Error Ci'].mask(m_gross_genre1_df['Bottom Error Ci'] < 0, m_gross_genre1_df['Mean Gross'])


# In[40]:


#making the barplot of Mean Worldwide Gross per Genre
plt.figure(figsize=(10, 5))

a = sns.barplot(x = 'Mean Gross', y = 'Genre', data = m_gross_genre1_df)

plt.errorbar(x = 'Mean Gross', y = 'Genre', xerr =[m_gross_genre1_df['Bottom Error Ci'], m_gross_genre1_df['Top Error Ci']],  data = m_gross_genre1_df, fmt = 'o', c = 'navy')

plt.title('Mean Worldwide Gross per Genre', color = 'black', fontsize = 18)
plt.xlabel('Mean Worldwide Gross', fontsize = 14)
plt.ylabel('Genre', fontsize = 14)

plt.savefig('mean_gross_genre_barplot.png')


# In[41]:


#2 sample T-test :Adventure - Action Genre Mean Worldwide Gross 
pd.options.display.float_format = '{:.6f}'.format
print('H0: There is no significant difference between adventure movies mean gross and action movies mean gross.\n ')

adventure_mean = m_gross_genre1_df[m_gross_genre1_df['Genre']=='Adventure']
adventure_mean = adventure_mean['Mean Gross']
action_mean = m_gross_genre1_df[m_gross_genre1_df['Genre']=='Action']
action_mean = action_mean['Mean Gross']

adventure_std = std_gross_genre1_df[std_gross_genre1_df['Genre']=='Adventure']
adventure_std = adventure_std['Std Gross']
action_std = std_gross_genre1_df[std_gross_genre1_df['Genre']=='Action']
action_std = action_std['Std Gross']

adventure_count = np.array(gross_genre_df[gross_genre_df['Genre1'] == 'Adventure'].count())[0]
action_count = np.array(gross_genre_df[gross_genre_df['Genre1'] == 'Action'].count())[0]


t = (adventure_mean - action_mean)/(np.sqrt((adventure_std**2 / adventure_count) + (action_std**2 / action_count)))

deg_freedom = adventure_count + action_count - 2
#ttest, pval = 2*(1 - stats.t.pdf(0.95, df = t))

print(t)

pd.options.display.float_format = '{:.1f}'.format


# In[42]:


#--------my data mining problem--------making Production Budget and Worldwide Gross Scatterplot
plt.figure(figsize=(10, 5))
sns.scatterplot(x = 'Budget', y='Gross', data = budget_gross_df, facecolor = 'skyblue', edgecolor ='darkblue')

plt.title('Production Budget - Worldwide Gross',color = 'black', fontsize = 18)

plt.xlabel('Production Budget', fontsize = 14)
plt.ylabel('Worldwide Gross', fontsize = 14)

plt.savefig('Budget_Gross_Scatterplot.png')


# In[43]:


#Pearson Correlation Coefficient: Production Budget and Worldwide Gross
pd.options.display.float_format = '{:.6f}'.format
budget_gross_df.corr(method = 'pearson')


# In[44]:


#Spearman Correlation Coefficient: Production Budget and Worldwide Gross
budget_gross_df.corr(method = 'spearman')


# In[45]:


#2 sample z-test1: Production Budget and Worldwide Gross
print('H0: Grosses are larger than budgets invested.\n ')

ztest, pval = stests.ztest(budget_gross_df['Gross'], x2 = budget_gross_df['Budget'], value = 0 , alternative = 'larger')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H0)')

else:
    
    print('\nAccept Null Hypothesis (H0)')


# In[46]:


#2 sample z-test2: Production Budget and Worldwide Gross
print('H1: Grosses are smaller than budgets invested.\n ')

ztest, pval = stests.ztest(budget_gross_df['Gross'], x2 = budget_gross_df['Budget'], value = 0 , alternative = 'smaller')
print('p-value: ', pval)

if pval<0.05:
    
    print('\nReject Null Hypothesis (H1)')

else:
    
    print('\nAccept Null Hypothesis (H1)')

pd.options.display.float_format = '{:.1f}'.format


# In[47]:


# making the DataFrame for the mean() RTRating and mean() IMDBRating per decade
ratings_dates_df = df[['RTRating', 'IMDBRating', 'Date']]
ratings_dates_df = ratings_dates_df.rename(columns = {'RTRating' : 'Rotten Tomatoes', 'IMDBRating': 'IMDB'})
ratings_dates_df = ratings_dates_df.groupby((ratings_dates_df.Date//10)*10).mean()

ratings_dates_df = ratings_dates_df[['Rotten Tomatoes', 'IMDB']]
ratings_dates_df.reset_index(level = 0, inplace = True)
ratings_dates_df = ratings_dates_df.rename(columns = {'Date' : 'Decade'})

ratings_dates_df = pd.melt(ratings_dates_df, id_vars = 'Decade', var_name = 'Website', value_name = 'Rating')
print(ratings_dates_df)


# In[48]:


#creating catplot for mean ratings per decade
plt.figure(figsize=(10, 5))
colors = ['darkred', 'black']
palette = sns.color_palette(colors)
sns.catplot(x = 'Decade', y = 'Rating', hue = 'Website', data = ratings_dates_df, palette = palette, kind = 'bar')

plt.title('Mean Ratings per Decade', color = 'black', fontsize = 18)

plt.savefig('ratings_decade_catplot.png')


# In[49]:


#creating pointplot for mean ratings per decade
plt.figure(figsize=(10, 5))
colors = ['darkred', 'black']
palette = sns.color_palette(colors)
sns.pointplot(x = 'Decade', y = 'Rating', hue = 'Website', data = ratings_dates_df, palette = palette, kind = 'point')

plt.title('Mean Ratings per Decade', color = 'black', fontsize = 18)

plt.savefig('ratings_decade_pointplot.png')


# In[50]:


#creating scatterplot for mean ratings per decade
plt.figure(figsize=(10, 5))
sns.heatmap(ratings_dates_df.pivot('Website', 'Decade', 'Rating'), cmap='Reds')

plt.title('Mean Ratings per Decade', color = 'black', fontsize = 18)

plt.savefig('ratings_decade_heatmap.png')

