#!/usr/bin/env python
# coding: utf-8

# # Analyzing video games sales patterns for the online store Ice.
# The online store Ice, sells video games all over the world. User and expert reviews, genres, platforms (e.g. Xbox or PlayStation), and historical data on game sales are available from open sources. We need to identify patterns that determine whether a game succeeds or not. 
# 

# ## Step 1. Open the data file and study the general information. 

# In[8]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[9]:


#Url Path for the data.
url='games.csv'

#open files
data_raw=pd.read_csv(url)
display(data_raw.head())
display(data_raw.tail())
display(data_raw.sample(10))


# In[10]:


#print general information of the dataframe
display(data_raw.info())


# In[11]:


display(data_raw.describe())
display(data_raw.describe(include=['object']))


# In[12]:


#data_raw.shape


# In[13]:


#Check unique values and their count in platform, genre,user_score and rating
data_raw['Platform'].value_counts()


# In[14]:


data_raw['Genre'].value_counts()


# In[15]:


data_raw['User_Score'].value_counts()


# In[16]:


data_raw['Rating'].value_counts()


# ### Conclusion:
# 1. There are many missing values in the year_of_release, users_score and critic_score.
# 2. Some missing values are also there in the name and genre.
# 3. There are also 0 values stored in NA, EU and Japan sales column. 
# 

# In[17]:


#Checking duplicate data 
data_raw[data_raw[['Name', 'Platform', 'Year_of_Release', 'Genre']].duplicated()]


# As shown above found 2 records as duplicate but in the first record name contain null value which is not valid so later we handle these type of record so now just need to check for 'Madden NFL 13' game.

# In[18]:


#checking all records for 'Madden NFL 13' game.
data_raw[(data_raw['Name']=='Madden NFL 13') & (data_raw.Platform=='PS3')]


# Deleting the data manually as it just one record for the index=16230.
# I also added the EU_sales figure of the deleting row to the previous data before deleting as the sales figure are important for our analysis. 

# In[19]:


#Deleting the data manually as it just one record for the index=16230.
#I also add the EU_sales figure in to the previous data before deleting the row where index=16230
eu_sales_temp=data_raw.iloc[604]['EU_sales']+data_raw.iloc[16230]['EU_sales']
data_raw.at[604,'EU_sales']=eu_sales_temp
print('New value updated in the column EU_sales for index =604--> ',data_raw.iloc[604]['EU_sales'])


# In[20]:


#Delete the row with the index =16230
try:
    data_raw = data_raw.drop(16230)
    print('Row deleted with the index :16230')
except:
    print('Row already deleted')
    


# ## Step 2. Prepare the data

# In[21]:


#Replace all column to lower case.
data_raw.columns=data_raw.columns.str.lower()

print(data_raw.columns)


# In[22]:


#counts no of 0 in all columns.
for i in data_raw.columns:
    print(i,len(data_raw[data_raw[i]==0]))
    


# In[23]:


#cheking for all null values.
data_raw.isnull().sum()
    


# There are 2 rows where name is null, we checked and find there is no way to fill these values and also these are only 2 rows so we delete them. 

# In[24]:


#Delete the rows where name is null.
#pd.set_option("display.max_rows", None)
print('Before deleting rows with NaN in name column:')
display(data_raw[(data_raw['name'].isnull())])
print('################################################################################################')
print()
data_raw.drop(data_raw[(data_raw['name'].isnull())].index, inplace=True)
print('After deleting rows with NaN in name column:')
display(data_raw[(data_raw['name'].isnull())])


# In[25]:


#pd.set_option("display.max_rows", None)
data_raw['year_of_release_temp']=data_raw[data_raw['year_of_release'].isnull()]['name'].apply(
                                                lambda x: [int(s) for s in x.split() if s.isdigit()]
)
#269 null rows

# there are some name column which contains year as well, so we can try to fetch the year and can update corresponding year_of_release 

display(data_raw[data_raw['year_of_release'].isnull()])


# In[26]:


#define function which will fetch the year part from the year_of_release_temp column.
def check_year(x):
    if x is np.nan:
        return x
    else:
        for i in x:
            if int(i>1900):
                return i
            else:
                continue


# In[27]:


data_raw['year_of_release_temp_clean']=data_raw['year_of_release_temp'].apply(check_year)

#update year from year_of_release_temp to year_of_release column.
data_raw['year_of_release'] = np.where((data_raw.year_of_release.isnull()), data_raw.year_of_release_temp_clean, data_raw.year_of_release)


# In[28]:


#display(data_raw[data_raw['year_of_release'].isnull()])
print(data_raw['year_of_release'].isnull().sum())
# After updation 254 rows still left. 


# In[29]:


#update rest rows with the median value on the basis of name column.
print('Before Updation:',data_raw['year_of_release'].isnull().sum())

data_raw['year_of_release']=data_raw.groupby('name')['year_of_release'].transform(
    lambda grp:grp.fillna(grp.median())
                          )#.agg('median')
print('After Updation :',data_raw['year_of_release'].isnull().sum())


# In[30]:


#still remaining 139 rows.
# So we will take most common year_of_release based on platform and genre and update the rest field with the mode value.
data_raw['year_of_release']=data_raw.groupby(['platform','genre'])['year_of_release'].transform(
    lambda grp:grp.fillna(grp.mode()[0])
                          )

print('After Updation count of null values in year_of_release :',data_raw['year_of_release'].isnull().sum())

#deleteing year_of_release_temp

data_raw.drop('year_of_release_temp', inplace=True, axis=1)
data_raw.drop('year_of_release_temp_clean', inplace=True, axis=1)


# In[31]:


print(data_raw.isnull().sum())


# In[32]:


data_raw.rating.value_counts()


# In[33]:


data_raw.rating.unique()
data_raw[data_raw.rating.isnull()]


# In[34]:


#creating dictionary for rating
rating_grouped=data_raw.groupby('genre')['rating'].agg(pd.Series.mode).reset_index()
rating_dic=dict(zip(rating_grouped.genre,rating_grouped.rating))
rating_dic


# In[35]:


print('Before Updation count of null vaues in rating:',data_raw['rating'].isnull().sum())
data_raw['rating']=data_raw['rating'].fillna(data_raw['genre'].map(rating_dic))
print('After Updation count of null values in rating:',data_raw['rating'].isnull().sum())


# In[36]:


#change types of the columns
data_raw.info()


# In[37]:


data_raw.isnull().sum()


# In[38]:


display(data_raw.describe())
display(data_raw.describe(include=['object']))


# In[39]:


print('Missing value % for user_score {0:.2f}%'.format(data_raw.user_score.isnull().sum()/len(data_raw)*100))
print('Missing value % for critic_score {0:.2f}%'.format(data_raw.critic_score.isnull().sum()/len(data_raw)*100))
#The missing value counts is very high for both the case. and we didn't find any way to fill those 
#so will leave as it is and move further.
#We will change the tbd also to null as it is also consider as null value.
#data_raw['user_score_new']=data_raw.user_score.replace('tbd', None)
data_raw['user_score']=data_raw.user_score.replace('tbd', None)

display(data_raw.describe(include=['object']))
print('After replacing tbd as null new % for missing value in user_score {0:.2f}%'.format(
    data_raw.user_score.isnull().sum()/len(data_raw)*100))


# 1. The missing value counts is very high for both the case. and we didn't find any way to fill those. So will leave as it is and move further.
# 2. I have also changed the tbd also to null as it is also consider as null value.

# In[40]:


data_raw.info()


# In[41]:


#pd.set_option("display.max_rows", 10)
#calculate total sales (the sum of sales in all regions) for each game
data_raw['total_sales']=data_raw[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
display(data_raw)


# In[42]:


data_raw.info()


# In[43]:


#Convert the data to the necessary types.
data_raw['name'] = data_raw['name'].astype('str')
data_raw['platform'] = data_raw['platform'].astype('str')
data_raw['year_of_release'] = data_raw['year_of_release'].astype('int')#year can contain only integer so changing the type to int.
data_raw['genre'] = data_raw['genre'].astype('str')
data_raw['rating'] = data_raw['rating'].astype('str')
data_raw['user_score'] = data_raw['user_score'].astype('float')
data_raw['critic_score'] = data_raw['critic_score'].astype('float')

data_raw.info()


# There are lots of data with 0 values in sales columm but we will not delete this as it may be possible that games were not sold in that region.

# #### Conclusion
# 1. First we checked all the data for null and 0 values. There is a lot of data which are not filled. So tried to figured out individual column how can we filled them..
# 2. name: There are only 2 rows for which data is not filled and as i can't figure how to fill them so deleted them from the dataframe.
# 3. platform: there are no null values found. 
# 4. year_of_release: There are lot of games for which year_of_release is not mentioned. When we checked the name column, I found with some name, year is also mentioned. So created one function to fetch the year from the name and updated the corresponding year in the relevant place. Even after that lot of data is still not updated and remanin as null. So for those we update those with the median value on the basis of name column. Still remains some rows so for those I took most common year_of_release based on platform and genre.
# 5. rating: Rating are updated based on genre. I used common rating based on the genre to update the rating column for null values.
# 6. User_Score and critic_Score. The missing value counts is very high for both the case and we didn't find any way to fill those. So will leave as it is and move further. I have also changed the tbd to null as it is also consider as null value.
# 7. Created new column as 'total_sales' which include total_sales from all region for each game.
# 9. There are lots of data available with 0 values in sales columm but we will not delete them as it may be possible that games were not sold in that region.
# 8. At the end updated the data type of required column (year_of_release, critic_score and user_score). 
# 
# So far we have checked all the data, updated some of the missing values, added some required column and changed the data type for columns. Now we move forward for analysing the data.
# 
# 
# 

# ### Step 3. Analyze the data

# #### Look at how many games were released in different years. Is the data for every period significant?

# In[44]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#for visualization purpose we will take only those records whose total_sales is greater than 0.
data_raw=data_raw[data_raw['total_sales']>0]


# In[45]:


#
data_games_byYear_pvt=data_raw.pivot_table(index='year_of_release', values='name', aggfunc={'name':'count'}).reset_index()
data_games_byYear_pvt.columns=['year','total no. of games']
data_games_byYear_pvt=data_games_byYear_pvt.sort_values(by='year')
ax=data_games_byYear_pvt.plot(title='Total Games Released by Year',x='year',y='total no. of games', grid=True,style='o')
ax.set_xlabel('Year')
ax.set_ylabel('Total Games Released')
plt.show()

#display(data_games_byYear_pvt)


# #### Conclusion
# Till 1995, sales are very low. May be because till that period video games were not that much popular or they are not available so easily. After that sales starts to increase till the year 2010 rapidly. After that sudden drop in sales and then continue to rise with nominal figure till 2015. In 2016 also sales decrease as compare to 2015.

# #### Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?

# In[46]:


#Showing platform wise total_sales .
data_platform_pvt=data_raw.pivot_table(index='platform', values='total_sales', aggfunc='sum').reset_index()
data_platform_pvt.columns=['platform','total_sales']

#saving top 6 platform for which sales are high.
df_topPlatform=data_platform_pvt.sort_values(by='total_sales',ascending=False).head(6)

ax=data_platform_pvt.sort_values(by='total_sales',ascending=False).plot(figsize=(10,5), x='platform',y='total_sales'
                                                                        ,kind='bar', title='Platform Wise Total Sales')
ax.set_xlabel('Platform')
ax.set_ylabel('Total Sales')
plt.show()


# In[47]:


ax=data_raw[data_raw['platform'].isin(df_topPlatform.platform)].pivot_table(
    index='year_of_release',columns='platform',values='total_sales',aggfunc='sum',fill_value=0
).plot(figsize=(12,7), title='Year wise total sales for top 6 platform')
ax.set_xlabel('Year of Release')
ax.set_ylabel('Total Sales - USD,Millions')
    
plt.show()


# #### Conclusion
# No or minimal sales available before 1995.<br/>
# 1. 'PS' platform sales starts to increase in 1994 and declined in 2004. Reaches to its highest level in 1997 or 1998 with 160 million dollars sales.
# 2. 'DS' platform sales starts to increase in 2003 and declined in 2013. Reaches to its highest level in 2006 or 2007 with 140 million dollars sales.
# 3. 'PS2' platform sales starts to increade in 1999 and declined in 2010. Reaches to its highest level in 2004 with 210 million dollars sales.
# 4. 'PS3' platform sales starts to increase in 2005 and declined in 2016. Reaches to its highest level in 2011 with 160 millions dollar sales.
# 5. 'Wii' platform sales starts to increase in 2005 and declined in 2013. Reaches to its highest level in 2010 with 210 million dollars sales.
# 6. 'X360' platform sales starts to increase in 2004 and declined in 2016. Reaches to its highest level in 2011 with 160 million dollars sales.<br/><br/>
# So from the above analysis, I can conclude that first we find top 6 platform based on their sales. After analzing year wise data for each of these platform it shows that normally for each platform now have zero sales. Almost all platform survive for 10 years and it reaches to its maximum sales after 4 or 5 years. After that it starts to decline.
# 

# #### Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017

# #### Work only with the data that you've decided is relevant. Disregard the data for previous years

# **Filltering the data.**
# We take data only after the year 2010 as we concluded before that In General platform needs 5-7 years  to reach to its maximum sales figure so now we will take data from last 5-6 years.
# 

# In[48]:


print('Before filter count',data_raw.shape)
#filltering the data. We take data only after the year 2010 as we concluded before that 
#In General platform needs 5-7 years  to reach to its maximum sales figure so now we will take data from last 5-6 years.
data_fil=data_raw[(data_raw['year_of_release']>2010)]
print('After filter count',data_fil.shape)


# #### Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms

# In[49]:


df=data_fil.pivot_table(index='year_of_release', columns='platform', values='total_sales', aggfunc='sum', fill_value=0).sort_values(by='year_of_release')
df
#df[['XOne', 'X360', 'WiiU','PS3','PS4','PC','3DS']]


# In[50]:


dynamic=df.T-df.T.shift(+1,axis=1)
dynamic
#dynamic[dynamic.index.isin(['XOne', 'X360', 'WiiU','PS3','PS4','PC','3DS'])]


# In[51]:


plt.figure(figsize=(13,9))
ax=sns.heatmap(dynamic,cmap='RdBu_r')
ax.set_title('Displaying Year Wise Potentially Profit of Platforms')
ax.set_xlabel('Year of Release')
ax.set_ylabel('Platform')


# #### Conclusion
# 
# The above graph shows that Potentially Profit of above Platforms. This is shown by the color of each cell against each platform.
# 1. The white color shows profit is same as of last year. The blue means that platform profit is declining and if the color moves towards red, it shows the platform is more profitable compare to last year.
# 2. The 3DS, WiiU,PC,PSP Platform sales were decreasing from last some years.
# 3. The Platform XOne, PS4 platform sales have just started to decrease drastically in 2016 but they come in 2013 so there are chances that they grow again in 2017.
# 4. **For X360, PS3 decrease density in sales in 2016 is less as compare to previous year while Wii, PSV, PSP sales are also less decreased in comparoson to the previous year but with very nominal value.**
# 5. PS2, DS are almost finished over the last years.
# 
# 

# #### Conclusion:
# 

# #### Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.

# In[52]:


data_fil.head(5)


# In[53]:


data_grouped=data_fil[data_fil['year_of_release']>2010].groupby(['platform','name'])['total_sales'].agg('sum').reset_index()
#data_grouped[data_grouped['platform']=='GB']
data_grouped=data_grouped[data_grouped.total_sales>0]
data_grouped


# Data is not much presentable. lots of outliers are plotted as the data is grouped by platform and name. Now we will group the data by platform and year and plot the box plot.

# In[54]:


data_grouped=data_fil[data_fil['year_of_release']>2010].groupby(['platform','year_of_release'])['total_sales'].agg('sum').sort_values(ascending=False).reset_index()
#data_grouped[data_grouped['platform']=='X360']
data_grouped


# In[55]:


plt.figure(figsize=(13,10))
ax=sns.boxplot(x='platform',y='total_sales',data=data_grouped)
ax.set_title('Displaying box plot for the global sales of all game')
ax.set_xlabel('Platform')
ax.set_ylabel('Total Sales - USD,Millions')


# #### Conclusion

# <s>1. The above Boxplot graph display the avg. total sales per year for each platform. From the above graph 'PS4' is the main platform for which median total_sales is also high.</s>
# 1. The above Boxplot graph display the total sales per year for each platform. The graph above shows that total_sales for 'PS3' is highest among all platform but the median of total_sales is high for 'PS4'.
# 2. Also there are some outliers. In this case outliers are special game which generate much revenue. So we can check for this kind of game and platform for future purpose.

# #### What about average sales on various platforms? 

# In[56]:


data_grouped=data_fil[data_fil['year_of_release']>2010].groupby(['platform','year_of_release'])['total_sales'].agg('mean').sort_values(ascending=False).reset_index()
#data_grouped[data_grouped['platform']=='X360']
data_grouped


# In[57]:


plt.figure(figsize=(13,10))
ax=sns.boxplot(x='platform',y='total_sales',data=data_grouped)
ax.set_title('Displaying box plot for the Average sales of all game')
ax.set_xlabel('Platform')
ax.set_ylabel('Total Sales - USD,Millions')


# If we see the average total_sales of platforms 'PS4' then we find avg. sales is high for PS4 as compare to other platforms.

# #### Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions

# In[58]:


#finding platform which has largest sales and no. of games compare to other platform.
df=data_fil[data_fil['year_of_release']>2010]
df.groupby('platform').agg({'total_sales':'sum', 'name':'count'}).sort_values(by=['total_sales','name'],ascending=False)
#platform=PS2


# <s>From the above table we found that PS2 have highest sales and number of games. So I will analyze this platform.</s><br/>
#     From the above table we found that PS3 have highest sales and number of games. So I will analyze this platform.

# In[59]:


#data_fil[data_fil['platform']=='PS2']


# In[60]:


df_scatter=data_fil[data_fil['platform']=='PS3']
display(df_scatter.head(5))


# In[61]:


df_scatter=df_scatter.dropna()


# In[62]:


df_scatter['user_score'].isnull().sum()


# In[63]:


df_scatter.info()


# In[64]:


df_scatter=df_scatter.sort_values(by='user_score')
df_scatter


# In[65]:


#creating function which will show scatter plot graph for each user_score and critic_score. 
def relation_sales_review(data,userScore):
    uniquePlat=data['platform'].unique()
    ax=data.plot(kind='scatter', x=userScore, y='total_sales' , style='o',grid=True,figsize=(10, 6),legend=False)# the lower the vehicle age the more price.
    plt.ylabel('total_sales')
    plt.xlabel(userScore)
    plt.title('Relationship between total_sales and '+userScore+' for the platform: '+str(uniquePlat))
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.show()
    print('Pearson correlation coefficient are: ', data[userScore].corr(data['total_sales']))


# In[66]:


#Draw scatter plot for user_score and critic_score.
relation_sales_review(df_scatter,'user_score')
relation_sales_review(df_scatter,'critic_score')


# In[67]:


#correlation between critic_score, total_score and total sales. 
df_scatter[['user_score','critic_score','total_sales']].corr()


# #### Conclusion
# 1. I choose PS3 platform for the analysis as it got highest no of sales and games. 
# 2. After analysing, we found there is no relationship between user_score and sales or critic_score and sales.
# 3. But found some positive relationship between critic_score and user_score.

# #### Keeping your conclusions in mind, compare the sales of the same games on other platforms

# In the above section, I found some platform for which sales have been increased in 2016. So now taking 3 of them for further analysis.

# In[68]:


#I am taking here below platform for this task.
topPlatfrm_list=['X360','PS3','Wii']
topPlatfrm_list


# In[69]:


#warnings.filterwarnings('ignore')
#data_fil.shape
#data_fil['platform'].unique()


# In[70]:


data_topGames=data_fil[(data_fil['platform'].isin(topPlatfrm_list)) &
        (data_fil.total_sales>0)].sort_values(by='total_sales',ascending=False)
data_topGames


# In[71]:


def display_scatter_graph_MultiplePlatform(data_full,PlatformList,columnName1,columnName2):
    plt.figure(figsize=(10, 6))
    for pltfrm in PlatformList:
        
        df=data_full[data_full['platform']==pltfrm]
        print('Pearson correlation coefficient between '+columnName1+' and '+columnName2+' for '+pltfrm+': ', 
              df[columnName1].corr(df[columnName2]))
        plt.scatter(x=df[columnName1], y=df[columnName2], marker='o',label=pltfrm)
   
    plt.legend()
    plt.xlabel(columnName1)
    plt.ylabel(columnName2)
    plt.title('Relationship between '+ columnName1+' and '+columnName2+' for Platform(s): '+str(PlatformList))
    plt.show() 
    


# In[72]:


#Displaying Scatter plot for multiple platform. It will show the relationship of platforms's total_sales with user_score, critic_score

display_scatter_graph_MultiplePlatform(data_topGames,topPlatfrm_list,'user_score','total_sales')
display_scatter_graph_MultiplePlatform(data_topGames,topPlatfrm_list,'critic_score','total_sales')


# 
data_topGames=data_fil[(data_fil['name'].isin(df_scatter['name'].unique())) &
        (data_fil.total_sales>0)].sort_values(by='total_sales',ascending=False)


#data_topGames=data_topGames.head(100)# take only top 100 games for platform.
data_topPlatform_byGames= data_topGames['platform'].unique()

display(data_topGames)
display(data_topPlatform_byGames)def display_scatter_graph(data_graph,columnName1,columnName2,title):
    ax=data_graph.plot(kind='scatter', x=columnName1, y=columnName2 , style='o'
                   ,grid=True,figsize=(5, 3),legend=False)# the lower the vehicle age the more price.
    ax.set_ylabel(columnName2)
    ax.set_xlabel(columnName1)
    ax.set_title('Relationship between total_sales and '+columnName1+' for the platform: '+title)
    print('Pearson correlation coefficient between '+columnName1+' and '+columnName2+' for '+title+': ', data_graph[columnName1].corr(data_graph[columnName2]))#display scatter plot and correlation coefficient between user_score, critic_score and sales.

for plt in data_topPlatform_byGames:
    df=data_topGames[data_topGames['platform']==plt]
    #df.dropna()
    display_scatter_graph(df,'user_score','total_sales',plt)
    display_scatter_graph(df,'critic_score','total_sales',plt)

  

# #### Conclusion
# 1. Found no such relationship between user_score and critic_score with platform(s).<br>
# <s>2. PS2: There is as -ve relation found between user_score and total_sales. thats strange.</s><br>
# <s>3. Wii: Found +ve relation ship but it's just a single record for this platform.</s>
# 

# #### Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?

# In[73]:


data_fil.pivot_table(
    index='genre',values='total_sales', aggfunc='sum').sort_values(
    by='total_sales',ascending=False).reset_index().plot(kind='bar',
    title='Total sales by Genre',x='genre',y='total_sales'
).set(xlabel='Genre', ylabel='Total Sales')
#display(data_fil_genre)


# In[74]:


import matplotlib.pyplot as plt
import numpy as np

data_fil_genre=data_fil.pivot_table(
    index='genre',values='total_sales', aggfunc='sum').sort_values(
    by='total_sales',ascending=False).reset_index()
data_fil_genre
plt.figure(figsize=(10, 10))
plt.pie(data_fil_genre['total_sales'], labels = data_fil_genre['genre'],autopct='%.2f%%')
plt.legend(title = "Genre")
plt.title('Genre wise distribution for Total Sales')
plt.show() 


# #### Conclusion
# 1. We can say now main profitable genre are 'Action', 'Sports', 'Shooter' and 'Role-playing'. So we can concentrate on this genre more.

# ## Step 4. Create a user profile for each region

# **For each region (NA, EU, JP), determine:**<br/>
# The top five platforms. Describe variations in their market shares from region to region.   

# In[75]:


# Top 5 platform for NA Sales
data_NA=data_fil[data_fil['na_sales']>0].pivot_table(index='platform',values='na_sales', aggfunc='sum').sort_values(by='na_sales', ascending=False).reset_index().head(5)
data_EU=data_fil[data_fil['eu_sales']>0].pivot_table(index='platform',values='eu_sales', aggfunc='sum').sort_values(by='eu_sales', ascending=False).reset_index().head(5)
data_JP=data_fil[data_fil['jp_sales']>0].pivot_table(index='platform',values='jp_sales', aggfunc='sum').sort_values(by='jp_sales', ascending=False).reset_index().head(5)

data_NA.plot(kind='bar',x='platform',y='na_sales', title='Top 5 platform in NA').set(xlabel='Platform', ylabel='Total NA Sales')
data_EU.plot(kind='bar',x='platform',y='eu_sales', title='Top 5 platform in EU').set(xlabel='Platform', ylabel='Total EU Sales')
data_JP.plot(kind='bar',x='platform',y='jp_sales', title='Top 5 platform in Japan').set(xlabel='Platform', ylabel='Total Japan Sales')


# <s>### Conclusion
# 1. DS, PS2 and PS3 are common in all three region. 
# 2. In NA, X360 is most popular platform with the total sales amount to approx.600 in USD million
# 3. In EU, PS2 is most popular with the total sales amount to approx.380 in USD million
# 4. While in Japan, DS is most popular with the total sales amount to approx.175 in USD million</s>

# ### Conclusion
# 1. PS3,PS4,3DS are common in all three region. 
# 2. In NA, X360 is most popular platform.
# 3. In EU, PS3 is most popular platform.
# 4. While in Japan, 3DS is most popular.

# #### The top five genres. Explain the difference

# In[76]:


def Display_Pie_Graph(data,yparam,lbl,region):
    plt.figure(figsize=(10, 5))
    plt.pie(data[yparam], labels = data[lbl],autopct='%1.0f%%')
    plt.legend(title = str(lbl).upper())
    plt.title(str(lbl).upper()+' wise distribution of sales in '+region)
    plt.show()


# In[77]:


#import matplotlib.pyplot as plt
data_NA=data_fil[data_fil['na_sales']>0].pivot_table(index='genre',values='na_sales', aggfunc='sum').sort_values(by='na_sales', ascending=False).reset_index().head(5)
data_EU=data_fil[data_fil['eu_sales']>0].pivot_table(index='genre',values='eu_sales', aggfunc='sum').sort_values(by='eu_sales', ascending=False).reset_index().head(5)
data_JP=data_fil[data_fil['jp_sales']>0].pivot_table(index='genre',values='jp_sales', aggfunc='sum').sort_values(by='jp_sales', ascending=False).reset_index().head(5)
Display_Pie_Graph(data_NA,'na_sales','genre','NA')
Display_Pie_Graph(data_EU,'eu_sales','genre','EU')
Display_Pie_Graph(data_JP,'jp_sales','genre','Japan')

#plt.figure(figsize=(10, 5))
#plt.pie(data_NA['na_sales'], labels = data_NA['genre'],autopct='%1.0f%%')
#plt.legend(title = "Genre")
#plt.title('Genre wise distribution in NA')
#plt.show()


# <s>#### Conclusion
# 1. Action, Sports and Misc are common genre which are amongst the top 5 genre in all region.
# 2. In NA and EU, Action and Sports are the top 2 genre.
# 3. While in Japan, Role-Playing is most popular genre.</s>

# #### Conclusion
# 1. Action, Role-Playing are common genre which are amongst the top 5 genre in all 3 region. 
# 2. In NA and EU, Action and Shooter are the top 2 genre.
# 3. While in Japan, Role-Playing is most popular genre.

# #### Do ESRB ratings affect sales in individual regions?

# In[78]:


data_fil.rating.unique()
data_fil.rating.value_counts()
data_fil.info()
#data_fil[data_fil['rating']=='']


# In[79]:


data_NA=data_fil[data_fil['na_sales']>0].pivot_table(index='rating',values='na_sales', aggfunc='sum').sort_values(by='na_sales', ascending=False).reset_index().head(5)
data_EU=data_fil[data_fil['eu_sales']>0].pivot_table(index='rating',values='eu_sales', aggfunc='sum').sort_values(by='eu_sales', ascending=False).reset_index().head(5)
data_JP=data_fil[data_fil['jp_sales']>0].pivot_table(index='rating',values='jp_sales', aggfunc='sum').sort_values(by='jp_sales', ascending=False).reset_index().head(5)
Display_Pie_Graph(data_NA,'na_sales','rating','NA')
Display_Pie_Graph(data_EU,'eu_sales','rating','EU')
Display_Pie_Graph(data_JP,'jp_sales','rating','Japan')


# <s>#### Conclusion
# 1. Games with rating 'E' and 'T' are most popular in all of the region.
# 2. Yes, we can say that ESRB rating affect the sales.</s>

# #### Conclusion
# 1. Games with rating 'E' and 'M' are most popular in NA and EU. While 'T' is mort popular in Japan.
# 2. Yes, we can say that ESRB rating affect the sales.
# 

# So now we can say that :
# 1. In NA region most perferred game gerne are 'Action' and 'Shooter' on the platform 'X360' with rating as 'M'.
# 2. In EU region most perferred game gerne are 'Action' and 'Shooter' on the platform 'PS3' with rating as 'M'.
# 2. In Japan region most perferred game gerne are 'Role-Playing' and 'Action' on the platform '3DS' with rating as 'T'.

# ## Step 5. Test the following hypotheses:

# #### Average user ratings of the Xbox One and PC platforms are the same.

# **Null Hypothesis H0:** Average user ratings of the Xbox One and PC platforms are the same<br/>
# **Alternative Hypothesis H1:** Average user ratings of the Xbox One and PC platforms are not same.

# In[80]:



data_fil['platform'].unique()


# In[81]:


#Test the hypotheses.
from scipy import stats as st
import numpy as np


data_Xbox=data_fil[data_fil['platform'].isin(['XOne'])].dropna()['user_score']
data_PC=data_fil[data_fil['platform']=='PC'].dropna()['user_score']

print('Variance of Xbox platform', data_Xbox.var())
print('Variance of PC Platform',data_PC.var())

alpha = .05 # critical statistical significance level
results = st.ttest_ind(
        data_Xbox, 
        data_PC,equal_var=False)#We pas equal_var as False as the variance of both sample are not equal.
print('p-value: ', results.pvalue)

if (results.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis") 

print('Avg. user rating of Xbox platform: {0}'.format(data_Xbox.mean()))   
print('Avg. user rating of PC platform: {0}'.format(data_PC.mean()))


# #### Conclusion
# After examine the p-value, We can say that we can't reject the null hypothesis, which implies average user rating from Xbox and PC platform are same.

# #### Average user ratings for the Action and Sports genres are different.

# **Null Hypothesis H0:** Average user ratings for the Action and Sports genres are same<br/>
# **Alternative Hypothesis H1:** Average user ratings for the Action and Sports genres are different.

# In[82]:


data_fil['genre'].unique()


# In[83]:


#Test the hypotheses.
from scipy import stats as st
import numpy as np

data_Action=data_fil[data_fil['genre']=='Action'].dropna()['user_score']
data_Sprt=data_fil[data_fil['genre']=='Sports'].dropna()['user_score']

print('Variance of Action genre', data_Action.var())
print('Variance of Sports genre',data_Sprt.var())

alpha = .05 # critical statistical significance level
results = st.ttest_ind(
        data_Action, 
        data_Sprt,equal_var=False)#We pas equal_var as False as the variance of both sample are not equal.
print('p-value: ', results.pvalue)


if (results.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis") 

print('Avg. user rating of Action genre: {0}'.format(data_Action.mean()))   
print('Avg. user rating of Sports genre: {0}'.format(data_Sprt.mean()))


# #### Conclusion
# <s>After examine the p-value, We can say that we can't reject the null hypothesis, which implies for the Action and Sports genres are same.</s>

# After examine the p-value, We can say that we reject the null hypothesis, which implies for the Action and Sports genres are different.

# ## Step 6. Write a general conclusion

# ### General Conclusion

# <s>1. These are platform which are still there in market. XOne, X360, WiiU,PS3,PS4,PC,PSV,3DS. Among these 3DS and XOne, PS4 and WiiU are not older than 4 to 5 years. So we can concentrate on these as I analyze that after 4 or 5 years platform reaches to its highest level. Also X360 is older but still surviving in the market so this is also one of the option.
# 2. While deciding the platform we should also consider the region. As in NA, X360 is most popular while in EU PS2 whereas in Japan DS the most popular platform.
# 2. User score and critic score does not affect the sales of platform but ESRB rating does.
# 3. Main profitable genre are 'Action', 'Sports', 'Shooter' and 'Role-playing'. Among these Action and Sports are most popular in NA and EU whereas Role-Playing is most popular in Japan.
# 4. Games with rating 'E' and 'T' are most popular in all of the region.
# 5. Average user rating from Xbox and PC platform are same.
# 6. Average user ratings for the Action and Sports genres are same.</s>

# 1. PS2, X360, PS3, Wii, DS, PS are the top 6 platform based on total_sales of all platform. Among these PS3, Wii, X360 are the platform which are still active in the market. 
# 2. After analzing year wise data for each of these platform it shows that normally for each platform now have zero sales. Almost all platform survive for 10 years and it reaches to its maximum sales after 4 or 5 years. After that it starts to decline. So based on this conclusion, I decided to take data after 2010. 
# 2. So After filtering the data, found these unique platforms 3DS,DS,PC,PS2,PS3,PS4,PSP,PSV,Wii,WiiU,X360,XOne.
# 3. From these DS, PS2 and PSP are out from the market now.
# 4. Sales are decreasing for all remaing in 2016. May be economy was not good in that year so gaming industry also face slow down in 2016.
# 5. The Platform XOne, PS4 platform sales have just started to decrease drastically in 2016 otherwise they are growing. Also i think that as they come in 2013 so there are chances that their sales grow in 2017.
# 6. Box plot shows that gobal sales are high for PS3, X360, PS4.
# 7. User score and critic score does not affect the sales of platform but ESRB rating does.
# 8. Top main profitable genre are 'Action', 'Sports', 'Shooter' and 'Role-playing'. 
# 9. In NA, X360 is most popular platform. In EU, PS3 and in Japan, 3DS is most popular.
# 10. In NA and EU, Action and Shooter are the top 2 genre. While in Japan, Role-Playing is most popular genre.
# 12. Games with rating 'E' and 'M' are most popular in NA and EU. While 'T' is mort popular in Japan.
# 13. Average user rating from Xbox and PC platform are same.
# 14. Average user ratings for the Action and Sports genres are different.<br/>
# 
# 
# So now from all the above point, I can say that:
# Pofit from the sales of games depends upon the region. So company first look the region and then should decide which game is to be launched.
# like for example:
# In NA region most perferred game gerne are 'Action' and 'Shooter' on the platform 'X360' with rating as 'M'.
# In EU region most perferred game gerne are 'Action' and 'Shooter' on the platform 'PS3' with rating as 'M'.
# In Japan region most perferred game gerne are 'Role-Playing' and 'Action' on the platform '3DS' with rating as 'T'
