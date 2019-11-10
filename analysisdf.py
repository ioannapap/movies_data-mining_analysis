import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy as sp
from scipy import stats

df = pd.read_csv('movies.csv')

#removing unnecessary columns
df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'US Gross', 'US DVD Sales', 'MPAA Rating', 'Running Time (min)', 'Distributor', 'Source', 'Creative Type'], axis = 1, inplace = True)

#renaming some columns
df.rename(columns = {'Worldwide Gross' : 'Gross', 'Production Budget' : 'Budget', 'Release Date' : 'Date', 'Major Genre' : 'Genre', 'Rotten Tomatoes Rating' : 'RT Rating'}, inplace = True)

#pruning unwanted rows from Gross
df = df[df.Gross != 'Unknown']
df = df[df.Gross != '0']

#fixing indices after pruning (starting from 1 instead of 0)
df.index = np.arange(1, len(df) + 1)

#fixing the format of Date
df.Date =  df['Date'].str[-2:]
