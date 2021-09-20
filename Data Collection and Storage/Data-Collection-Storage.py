#!/usr/bin/env python
# coding: utf-8

# ## Step 4. Exploratory data analysis (Python)

# ### Importing the files

# In[35]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#import requests# # Import the library for sending requests to the server
#import re## importing the library for regular expression
#from bs4 import BeautifulSoup # Import the library for webpage parsing

pd.set_option("display.max_rows", 10)


# In[36]:


#Url Path for the data.
url1='datasets/project_sql_result_01.csv'
url2='datasets/project_sql_result_04.csv'

#Loading files locally in to dataframe
data_company=pd.read_csv(url1)
data_dropoff_location=pd.read_csv(url2)


# ### Verifying data and their types.

# In[37]:


#Verifying files and data.
display(data_company.head())
display(data_dropoff_location.head())


# In[38]:


display(data_company.tail())
display(data_dropoff_location.tail())


# In[39]:


display(data_company.sample(5))
display(data_dropoff_location.sample(5))


# In[40]:


#Verifying basic informatoin and data type of company table.
data_company.info()


# In[41]:


#Verifying basic informatoin and data type of dropoff table.
data_dropoff_location.info()


# In[42]:


#checking the table for null values.
print(data_company.isnull().sum())


# In[43]:


#checking the table for null values.
data_dropoff_location.isnull().sum()


# #### Conclusion
# It's seems data types of both the tables are correct. Also tables does not contain null values. So we move to further steps.

# ### Study the data they contain

# In[44]:


#checking the data in data_company table.
display(data_company.describe())
display(data_company.describe(include=['object']))
data_company['company_name'].unique()


# In[45]:


#checking the data in data_dropoff_location table.
display(data_dropoff_location.describe())
display(data_dropoff_location.describe(include=['object']))
#data_company['company_name'].unique()


# In[46]:


#Checking duplicate records.
print(data_company['company_name'].duplicated().sum())
print(data_dropoff_location['dropoff_location_name'].duplicated().sum())


# #### Conclusion:
# 1. It seems there is a huge variation in the data.
# 2. There are approx. 50% of data for which trips_amount is below 200. We can exclude all those later for further analysis.
# 3. As 75% of trip_amount of individual taxi reaches only till approx. 2000 so we can say now there are very few taxi companies which are holding the maximum market.
# 4. When I see the dropoff data, It's revealed that there are only few dropoff location for which there are high demand. so we can concentrate only those area initially.

# So far We have checked both the tables for null, 0 and duplicates values. Their Data types are also valid in both the table. So now we move forward for further analysis.

# ### Identify the top 10 neighborhoods in terms of drop-offs

# In[47]:


data_dropoff_location_Top10 =data_dropoff_location.sort_values(by='average_trips', ascending=False).head(10)


# In[48]:


display(data_dropoff_location_Top10)


# ### Make graphs: taxi companies and number of rides, top 10 neighborhoods by number of dropoffs

# In[49]:


#Importing library for graph.
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# **As we notcied above, there are many taxi companies for which trips_amount are even less than 100. SO we filter some of those and than make Graph.**
# 1. We first sort all data by trip_amount and take data of first 30 records and group all others in 'Others' group and take mean value for the same. So after that we will have 31 records.

# In[50]:



data_company_grp= data_company.sort_values(by='trips_amount', ascending=False)[31:]#fetching all records from 31 index.

data_company_grp['company_name']='Others'
data_company_grp['trips_amount']=(data_company_grp['trips_amount'].mean())
data_company_grp.drop_duplicates(subset=['company_name','trips_amount'],inplace=True)#Creates df with 'Others' and their mean value.
#display(data_company_grp)

data_company=data_company.sort_values(by='trips_amount', ascending=False)[0:31]
data_company=data_company.append(data_company_grp,ignore_index=True,sort=True)#Combine two df.
display(data_company)#Final data.


# In[51]:


#Plotting Bar Graph for the above df.
fig, ax = plt.subplots(figsize=(15,10), facecolor='white', dpi= 80)
ax.vlines(x=data_company.company_name, ymin=0, ymax=data_company.trips_amount, color='firebrick', alpha=0.7, linewidth=20)

# Title, Label, Ticks and Ylim
ax.set_title('Taxi companies and their total trips', fontdict={'size':22})
ax.set_ylabel('Total Trips')
ax.set_xlabel('Company Name')
plt.xticks(data_company.company_name, data_company.company_name, rotation=60, horizontalalignment='right', fontsize=12)
plt.show()


# #### Conclusion
# 1. As there are lost of data of small trips so I take the first 30 highest taxi companies based on their total trips and rest are grouped in the Other category.
# 2. Flash Cab has the highest no. of trips from all other taxi companies. 
# 3. Also the other taxi company are far behind than Flash Cab. So we need some more data from Flash Cab like their taxi types, No. of taxis and their availabily etc. to understand why their total trips are the highest.

# ### Top 10 neighborhoods by number of dropoffs

# In[52]:


#Plotting Lollipop graph for dropoff location.
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=data_dropoff_location_Top10.index, ymin=0, ymax=data_dropoff_location_Top10.average_trips, 
          color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=data_dropoff_location_Top10.index, y=data_dropoff_location_Top10.average_trips,
           s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Top 10 neighborhoods by number of dropoffs', fontdict={'size':22})
ax.set_ylabel('Average Trips')
ax.set_xticks(data_dropoff_location_Top10.index)
ax.set_xticklabels(data_dropoff_location_Top10.dropoff_location_name.str.upper(), rotation=60, 
                   fontdict={'horizontalalignment': 'right', 'size':12})
# Annotate
for row in data_dropoff_location_Top10.itertuples():
    ax.text(row.Index, row.average_trips+.5, s=round(row.average_trips, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)
plt.show()


# ### Conclusion:

# Loop, River North neighborhood are highest in terms of dropoffs.<br>
# 
# So we can conclude that, we can include these dropoffs location in our taxi comapany initially, as they are the highest dropoffs location in Chicago.
# 

# ## Step 5. Testing hypotheses (Python)

# In[53]:


#Import packages
from scipy import stats as st
import numpy as np


# In[54]:


#Url Path for the data.
url3='https://code.s3.yandex.net/datasets/project_sql_result_07.csv'

#Loading files locally in to dataframe
data_LoopToHare=pd.read_csv(url3)
display(data_LoopToHare)


# ### "The average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays."

# **Null Hypothesis H0:** Average duration of rides from the Loop to O'Hare International Airport does not changes on rainy Saturdays<br/>
# **Alternative Hypothesis H1:** Average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays.

# In[55]:


data_LoopToHare=data_LoopToHare.dropna()# droping all null values.

data_LoopToHare_nonrainy=data_LoopToHare[data_LoopToHare['weather_conditions']=='Good']['duration_seconds']# fetching only non rainy day data.
data_LoopToHare_rainy=data_LoopToHare[data_LoopToHare['weather_conditions']=='Bad']['duration_seconds']#fetching only rainy data.

print('Variance of duration of non rainy saturday ', data_LoopToHare_nonrainy.var())
print('Variance of duration of rainy ',data_LoopToHare_rainy.var())

alpha = .05 # critical statistical significance level
results = st.ttest_ind(
        data_LoopToHare_nonrainy, 
        data_LoopToHare_rainy,equal_var=False)#We pas equal_var as False as the variance of both sample are not equal.
print('p-value: ', results.pvalue)

if (results.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis") 

        
#converting avg. seconds in to minutes.
avg_nonrainy=(data_LoopToHare_nonrainy.mean()/60)
avg_rainy=(data_LoopToHare_rainy.mean()/60)
print('Avg. duration of non rainy Saturdays: {0:.2f}'.format(avg_nonrainy))   
print('Avg. duration of rainy Saturdays: {0:.2f}'.format(avg_rainy))


# As we have two statistical populations here which are based on same samples so we apply the method scipy.stats.ttest_ind(array1, array2, equal_var).
# 
# After examine the p-value, We can say that we reject the null hypothesis, So now we can conclude that "Average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays".
# 
# I also calculated the average for both rainy and non rainy saturday and found that on an average taxi arrives 7 min. faster on non rainy saturday than on rainy saturday.
# 

# In[ ]:




