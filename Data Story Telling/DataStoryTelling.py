#!/usr/bin/env python
# coding: utf-8

# # Project Description:

# Together with my friend I decided to open a small robot-run cafe in Los Angeles. This project is promising but expensive, so me and my partners decide to try to attract investors. They’re interested in the current market conditions. As I am an analytics person so my partners have asked me to prepare some market research. I have got open-source data on restaurants in LA for analyzing.

# ## Step 1. Download the data and prepare it for analysis

# In[2]:


# #installing libraryies.
get_ipython().system('pip install -q usaddress')
# !pip install plotly
get_ipython().system('pip install --upgrade -q seaborn')


# In[1]:


#Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime as dt
import warnings

import seaborn as sns
import plotly.express as px 
from plotly import graph_objects as go

#importing library for parsing address
import usaddress

warnings.filterwarnings('ignore')
#pd.set_option("display.max_rows", 10)
#pd.set_option('max_colwidth', 400)


# In[4]:


#Url Path for the data.
la_Data_url='rest_data_us.csv'

#Loading files locally in to dataframe
data_restLA=pd.read_csv(la_Data_url)

#Verifying files and data.
display(data_restLA.head())
display(data_restLA.tail())
display(data_restLA.sample())


# In[5]:


#Checking general information
data_restLA.info()


# All data types seems correct.

# In[6]:


data_restLA.shape


# In[7]:


#Checking null columns
data_restLA.isnull().sum()


# Chain column got 3 null fileds.

# In[8]:


data_restLA[data_restLA.chain.isnull()]


# In[9]:


data_restLA[data_restLA.object_name.str.contains('TAQUERIA LOS',na=False)]
#str.contains('sh|rd', regex=True, na=True)]


# In[10]:


data_restLA[data_restLA.object_name.str.contains('JAMMIN',na=False)]


# In[11]:


data_restLA[data_restLA.object_name.str.contains('THEATER',na=False)]


# Checked all the related data for all those object whose chain contains null values but found no relationship so updating chain value as False where it is null.

# In[12]:


#updating all null values to False in chain columns.
data_restLA["chain"].fillna(False, inplace = True)


# In[13]:


#Cheking for invaid data in the columns.
display(data_restLA[data_restLA['number']<=0])


# In[14]:


#Checking duplicates rows.
#'chain','object_type','number'
data_restLA.duplicated(subset=['object_name','address']).sum()


# In[15]:


data_restLA.describe(include='all')  


# In[16]:


data_restLA.object_name.value_counts()


# In[17]:


data_restLA.address.value_counts()


# In[18]:


data_restLA.chain.value_counts()


# In[19]:


data_restLA.object_type.value_counts() 


# #### Conclusion:
# There are some data in object_name and address which contains some invalid character/data in the field. So first will correct them and then perform the analysis.

# In[20]:


#create function to clean the object/establishment name.
def ValidateObjectName(ObjectName):#ObjectNames
    name=ObjectName
  
    #for name in data_restLA.object_name:#ObjectNames:
    if ('#' in name):
        name=name.split('#')[0].strip()            
    if ('-' in name):
        tmp = name.split('-')
        if (len(tmp)>=2):
            if (tmp[1].isdecimal()):
                name = tmp[0].strip()
    #print(name)
    return name
 


# In[21]:


data_restLA['object_name_clean']=data_restLA.object_name.apply(ValidateObjectName)


# In[22]:


#creating function to fetch the correct address from the raw address field.
def ValidateAddress(rawAddress):
    #cleanAddress=rawAddress
    if rawAddress.startswith('OLVERA'):
        cleanAddress = 'OLVERA, Los Angeles, USA'
    elif rawAddress.startswith('1033 1/2 LOS ANGELES ST'):
        cleanAddress = '1033 1/2 LOS ANGELES ST, Los Angeles, USA'
    else:
        raw_address=usaddress.parse(rawAddress)
        dict_address={}
        for i in raw_address:
            dict_address.update({i[1]:i[0]})
        if 'StreetNamePostType' in dict_address:
            cleanAddress = dict_address['AddressNumber'] + " " + str(dict_address['StreetName']) +                 " " + str(dict_address['StreetNamePostType'])+str(', Los Angeles, USA')
        else:
            cleanAddress = dict_address['AddressNumber'] + " " + str(dict_address['StreetName']) +                 " "+str(', Los Angeles, USA')
            
        #cleanAddress=dict_address['AddressNumber']+" "+str(dict_address['StreetName'])+str(', Los Angeles,USA')
        
    return cleanAddress


# In[23]:


data_restLA['address_clean']=data_restLA.address.apply(ValidateAddress)


# ## Step 2. Data analysis

# ### 1. Investigate the proportions of the various types of establishments. Plot a graph.

# In[25]:


objectwise_data=data_restLA.groupby('object_type')['id'].count()
objectwise_data=objectwise_data.reset_index()
objectwise_data


# In[26]:


name_rest =objectwise_data.object_type
values =objectwise_data.id
fig = go.Figure(data=[go.Pie(labels=name_rest, values=values,title='Types of Establishments')])

fig.show() 


# #### Conclusion:
# From the above graph, we can conclude that restaurant type establishment have the mojority of proportions i.e approx. 75% are Restaurants, 11% are Fast food, 5% Cafe and so on.

# ### 2. Investigate the proportions of chain and nonchain establishments. Plot a graph.

# In[27]:


chainwise_data=data_restLA.groupby('chain')['id'].count()
chainwise_data=chainwise_data.reset_index()
chainwise_data.loc[chainwise_data.chain==False,'Desc']='Nonchain'
chainwise_data.loc[chainwise_data.chain==True,'Desc']='Chain'
chainwise_data


# In[28]:


name =chainwise_data.Desc
values =chainwise_data.id
fig = go.Figure(data=[go.Pie(labels=name, values=values,title='Chain/NonChain proportions of Establishments')],
                )
fig.show() 


# Around 62% are non-chain and 32% are chain establishments.

# ### 3. Which type of establishment is typically a chain?

# In[29]:


chainwise_object=data_restLA.groupby(['object_type','chain'])['id'].agg('count')
chainwise_object=chainwise_object.reset_index()
chainwise_object.loc[chainwise_object.chain==False,'chainType']='Nonchain'
chainwise_object.loc[chainwise_object.chain==True,'chainType']='Chain'
chainwise_object


# In[30]:


fig = px.bar(chainwise_object, x='object_type', y='id', title='Displaying Establishments by Chain/Non-chain wise'
             ,color='chainType',labels={
                     "object_type": "Establishment Type",
                     "id":"Establishments count"
                 })
#fig.update_xaxes(tickangle=45)
fig.show()


# #### Conclusion:
# We can see here Bakrey is typically a chain establishment.

# ### 4. What characterizes chains: many establishments with a small number of seats or a few establishments with a lot of seats?

# In[31]:


sns.set(style="darkgrid")
fig_dims = (10, 5)
fig, axs = plt.subplots(figsize=fig_dims)
sns.histplot(data=data_restLA[data_restLA.chain==False], x="number", color="skyblue", label="NonChain", kde=True,ax=axs)
sns.histplot(data=data_restLA[data_restLA.chain==True], x="number", color="red", label="Chain", kde=True,ax=axs)
axs.set(xlabel='Seats count', ylabel='Establishment count',title='Comparison between seats count and establishments count') 
axs.legend() 
fig.show()


# #### Conclusion:
# As we can see from the above graph, most of the establishment have less than 50 seats. Also there are more nonchain type establishment as compare to chain type. So we can conclude that chain type have many establishments with a small number of seats and very less establishments with a lots of the seats. 

# ### 5. Determine the average number of seats for each type of restaurant. On average, which type of restaurant has the greatest number of seats? Plot graphs.

# In[31]:


seatWise_object=data_restLA.groupby('object_type')['number'].agg('mean')
seatWise_object=seatWise_object.reset_index()
seatWise_object.rename(columns={'number':'AverageSeats'}, inplace=True)


# In[32]:


seatWise_object.sort_values(by='AverageSeats', ascending=True, inplace=True)
seatWise_object


# In[33]:


fig = px.bar(seatWise_object, x='object_type', y='AverageSeats', 
             title='Displaying average seats restaurant basis', color='object_type',labels={
                     "object_type": "Establishment Type",
                     "AverageSeats": "Average no. of Seats"
                 })
#fig.update_xaxes(tickangle=45)
fig.show()


# #### Conclusion:
# From the above graph, we can conclude that Restaurant type have highest no. of seats on an average.

# ### 6. Put the data on street names from the address column in a separate column.

# In[37]:


#creating function to fetch the street name from the address.
def StreetName(raw):
    if raw.startswith('OLVERA'):
        street_name='OLVERA'
    else:
        raw_address=usaddress.parse(raw)
        street_name_list=[]
        for i in raw_address:
            if (i[1]=='StreetName' or i[1]=='StreetNamePostType'):
                street_name_list.append(str(i[0]).replace(',',''))
        if len(street_name_list)>0:
            street_name=' '.join(street_name_list)
        else:
            street_name=''
    return street_name


# In[38]:


#data_restLA.address.head(10).apply(StreetName)


# In[39]:


#adding the street name column to the table. 
data_restLA['street_name']=data_restLA.address_clean.apply(StreetName)


# In[40]:


data_restLA[data_restLA.street_name=='']


# In[41]:


data_restLA.sample(5)


# ### 7. Plot a graph of the top ten streets by number of restaurants.

# **Note**: considering restaurants as establishments.

# In[42]:


#streetwise_object=data_restLA.groupby('street_name')['id'].agg('count').reset_index()
streetwise_object=data_restLA.groupby('street_name').agg({'id':'count','number':['sum','mean']}).reset_index()
streetwise_object.columns=['street_name','object_count','seats_count','avg_seats']
streetwise_object.rename(columns={'id':'object_count'},inplace=True)
streetwise_object.sort_values(by='object_count', ascending=False,inplace=True)
streetwise_object_top10=streetwise_object.head(10)


# In[43]:


streetwise_object_top10


# In[44]:


print('Average of top 10 streets are, each street have approx. {0:.0f} establishments and each establishment have {1:.2f} seats.'.format(streetwise_object_top10.object_count.mean(),streetwise_object_top10.avg_seats.mean()))


# In[45]:


fig = px.bar(streetwise_object_top10, x='street_name', y='object_count', 
             title='Displaying top ten streets by number of establishments', color='street_name',labels={
                     "street_name": "Street Name",
                     "object_count": "No. of Establishments"
                 })
#fig.update_xaxes(tickangle=90)
fig.show()


# ### 8. Find the number of streets that only have one restaurant.

# In[46]:


count=len(streetwise_object[streetwise_object.object_count==1])
print('There are {0} streets that only have one establishment.'.format(count))


# ### 9. For streets with a lot of restaurants, look at the distribution of the number of seats. What trends can you see?

# In[47]:


#streetwise_object_top10['seat_per_establishment']=streetwise_object_top10.seats_count/streetwise_object_top10.object_count
streetwise_object_top10


# In[48]:


plt.figure(figsize=(20,10))
axs=sns.displot(data=streetwise_object_top10, x="avg_seats", kind="kde")
axs.set(xlabel='Seats count', 
        title='Distribution of the number of seats for top 10 streets.') 
#axs.legend() 
plt.show()


# In[49]:


print('On an average top 10 streets have {0:.2f} seats.'.format(streetwise_object_top10.avg_seats.mean()))


# #### Conclusion:
# The distribution is normal which implies that on an average each establishment have 46 seats in top 10 streets.

# ## Overall conclusion

# After exploring the data on restaurants in LA, I notice/analyze the below points:
# 1. Among all types of establishments, restaurants are the most popular type of establishment.
# 2. There are more nonchain type of establishments as compare to chain type.
# 3. For chain type, Bakery, Fast Food or Cafe are the mostly favoured while for non chain type Restaurants, Bar or Pizza are mostly favoured.
# 4. Chain type are characterizes by the fact "many establishments with a small number of seats".
# 5. On an average each Restaurants have 48 seats, Bar 45, Fast Food 32, Pizza 28, cafe 25 and Bakery 21 seats.
# 6. Top 10 streets are SUNSET BLVD,WILSHIRE BLVD, PICO BLVD, WESTERN AVE, FIGUEROA ST, OLYMPIC BLVD, VERMONT AVE, MONICA BLVD, 3RD ST and HOLLYWOOD BLVD.
# 7. If we see average of these top 10 streest, we found each street have approx. 324 establishments and each establishment have 46 seats on an average.<br>
# So all these points suggests that we can go for non chain type like Restaurant or Bar. We can open this on any of top 10 streets as mentioned above with a capacity of approx.46 seats.
# 

# ##  Step 3. Presentation

# Presentation: https://drive.google.com/file/d/17OLz_O5cxs68uqWO4DYDzYIc8HmlLVJ4/view?usp=sharing

# In[ ]:




