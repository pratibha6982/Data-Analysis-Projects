#!/usr/bin/env python
# coding: utf-8

# ## Step 1. Download the data and prepare it for analysis
# (Exploratory data analysis)

# ### Download all data locally

# In[111]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", 10)


# In[112]:


#Url Path for the data.
url1='visits_log_us.csv'
url2='orders_log_us.csv'
url3='costs_us.csv'

#Loading files locally in to dataframe
data_visits=pd.read_csv(url1)
data_orders=pd.read_csv(url2)
data_costs=pd.read_csv(url3)


# ### Verifying data and their types.

# In[113]:


#Verifying files and data.
display(data_visits.head())
display(data_orders.head())
display(data_costs.head())


# In[114]:


display(data_visits.tail())
display(data_orders.tail())
display(data_costs.tail())


# In[115]:


#Verifying basic informatoin and data type of all tables.
data_visits.info(memory_usage='deep')
data_orders.info(memory_usage='deep')
data_costs.info(memory_usage='deep')


# 1. Found some data for which data type can be changed like for Start Ts and End Ts data type should be datetime. 
# 2. Before changing the data type we first change all column name to lower case and also replace all space in coulmn names with '_'  character.

# In[116]:


def update_columns_name(df):
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    print(df.columns)


# In[117]:


update_columns_name(data_visits)
update_columns_name(data_orders)
update_columns_name(data_costs)


# In[118]:


#checking the table for null values.
print(data_visits.isnull().sum())
print(data_orders.isnull().sum())
print(data_costs.isnull().sum())


# In[119]:


#checking the table for duplicates values.
print(data_visits.duplicated().sum())
print(data_orders.duplicated().sum())
print(data_costs.duplicated().sum())


# Found no null and no duploicates values. 

# ### Study the data they contain.

# In[120]:


#checking the data in data_visits table.
display(data_visits.describe())
display(data_visits.describe(include=['object']))


# In[121]:


data_visits['device'].unique()


# Device column contain only 2 unique value so we can change the data type of this column to 'category'.

# In[122]:


#checking the data in data_orders table.
display(data_orders.describe())
display(data_orders.describe(include=['object']))
#data_visits['device'].unique()
data_orders.info()


# ### Changing data type.

# In[123]:


#Changing data type in visit table.
data_visits.info(memory_usage='deep')
data_visits['start_ts'] =  pd.to_datetime(data_visits['start_ts'], format="%Y-%m-%d %H:%M:%S")
data_visits['end_ts'] =  pd.to_datetime(data_visits['end_ts'], format="%Y-%m-%d %H:%M:%S") 
data_visits['device'] = data_visits['device'].astype('category') 
data_visits.info(memory_usage='deep')


# In[124]:


#Changing data type in order table.
data_orders.info(memory_usage='deep')
data_orders['buy_ts'] =  pd.to_datetime(data_orders['buy_ts'], format="%Y-%m-%d %H:%M:%S")
data_orders.info(memory_usage='deep')


# In[125]:


#added code v.1
#Changing data type in costs table.
data_costs.info(memory_usage='deep')
data_costs['dt'] =  pd.to_datetime(data_costs['dt'], format="%Y-%m-%d %H:%M:%S")
data_costs.info(memory_usage='deep')


# In[126]:


#added code v.1
display(data_visits['end_ts'].max())


# As some values are more than May'2018 so we discard all the data greater than May'2018.

# In[127]:


#added code v.1
#сheck no of the sessions which have greater time frame than the specified time interval.
data_visits[data_visits['end_ts'].dt.date>pd.to_datetime('2018-05-31')]


# In[128]:


#added code v.1
#fetch only those data which matches the specified time interval.
data_visits=data_visits[data_visits['end_ts'].dt.date<pd.to_datetime('2018-06-01')]
#Verify the data max date limit
data_visits[data_visits['end_ts'].dt.date>pd.to_datetime('2018-05-31')]


# ### Conclusion:

# So far we have checked all the tables, their data and updated the data types wherever required. So now we move to next part to analyse these data and prepare the report. 

# ## Step 2. Make reports and calculate metrics

# ### Product

# ### - How many people use it every day, week, and month?

# In[129]:


#To make analysis for every day, week and month, first we fetch year, month and week from the visit table.

data_visits['start_ts_year']  = data_visits['start_ts'].dt.year
data_visits['start_ts_month'] = data_visits['start_ts'].dt.month
data_visits['start_ts_week']  = data_visits['start_ts'].dt.week
data_visits['start_ts_date'] = data_visits['start_ts'].dt.date
print(data_visits.head()) 


# In[130]:


DAU = data_visits.groupby('start_ts_date').agg({'uid': 'nunique'})
WAU = data_visits.groupby(['start_ts_year', 'start_ts_week']).agg({'uid': 'nunique'}).reset_index()
MAU = data_visits.groupby(['start_ts_year', 'start_ts_month']).agg({'uid': 'nunique'}).reset_index()


print(DAU)
print(WAU)
print(MAU) 


# In[131]:


DAU_total=DAU.uid.mean()
WAU_total=WAU.uid.mean()
MAU_total=MAU.uid.mean()

sticky_wau=DAU_total/WAU_total*100
sticky_mau=DAU_total/MAU_total*100

print('On an Average Daily {0:.0f} users visits to Yandex.Afisha'.format(DAU_total))
print('On an Average Weekly {0:.0f} users visits to Yandex.Afisha.'.format(WAU_total))
print('On an Average Monthly {0:.0f} users visits to Yandex.Afisha.'.format(MAU_total))

print('Sticky factor WAU',round(sticky_wau))
print('Sticky factor MaU',round(sticky_mau))


# In[132]:


# Line graph for DAU.
#import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(DAU.reset_index().start_ts_date,DAU.uid ,color='tab:red')
plt.title('Daily Activity User Graph')
plt.xlabel('Date-Month')
plt.ylabel('Count of users')
plt.show()


# In[133]:


#print(MAU.columns)

MAU['Year-Month']=+MAU['start_ts_year'].astype(str)+'-'+MAU['start_ts_month'].astype(str)
print(MAU)


# In[134]:


#Bar Graph for MAU.
#plt.figure(figsize=(10,5))
MAU.plot(kind="bar", title='Monthly Active User',x='Year-Month', y='uid', figsize=(10,5))
plt.xlabel("Year-Month")
plt.ylabel("Total User")


# #### Conclusion:
# 1. After looking at DAU and MAU, we can say that users visits are maximum in November and December. 
# 2. Visists are at least in june, july, august and september.
# 3. As per my analyses, Users spent more time on the website in winter and spring season.

# 

# In[135]:


#WAU
WAU['Year-Week']=+WAU['start_ts_year'].astype(str)+'-'+WAU['start_ts_week'].astype(str)
print(WAU)


# In[136]:


# Line graph for WAU.
#import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.plot(WAU['Year-Week'],WAU.uid ,color='tab:red')
plt.title('Weekly Activity User Graph')
plt.xlabel('Year-Week')
plt.ylabel('Count of users')
plt.xticks(rotation=90)
plt.show()


# #### Conclusion v.1
# 1. As per above graphs, Users spends more time in winter and spring, This could be because website sells something related to winter/spring or something like movie tickets becuase it is indoor activity and in winter people generally prefer to go for indoor activity like theatre. 
# 2. If we look at daily graph, in March users count reaches to zero, may be that time some technical issue arrise so nobody was able to login in to the website.
# 3. In the weekly graph, There is sharp increase in 2017-46, 2017-47 and 2017-48 week and decline in 2018-12 week.

# ### - How many sessions are there per day? (One user might have more than one session.)

# In[137]:


#caculating Session per day

sessions_per_day=data_visits.groupby(['start_ts_date']).agg({'uid':['count','nunique']})
sessions_per_day.columns = ['n_sessions', 'n_users']
sessions_per_day['average_session_per_user']=sessions_per_day['n_sessions']/sessions_per_day['n_users']
sessions_per_day=sessions_per_day.sort_values(by='n_sessions',ascending=False)
sessions_per_day


# In[138]:


kwargs=dict(alpha=0.5,bins=30)
plt.hist(sessions_per_day['n_sessions'],**kwargs, label='Sessions')
plt.gca().set(title='Session Per Day')
plt.ylabel('No. Of Session') 
plt.xlabel('Count') 
plt.legend()
plt.show()


# In[139]:


#code added v.1
kwargs=dict(alpha=0.5,bins=30)
plt.hist(sessions_per_day['average_session_per_user'],**kwargs, label='Avg. Sessions Per User')
plt.gca().set(title='Avg. Session Per User')
plt.ylabel('Count') 
plt.xlabel('No. Of Session Per User') 
plt.legend()
plt.show()


# As we can see from the above graph, data is not normally distributed and also there are some outlier available at 4000 so now I'll take median value to find the average session length.

# In[140]:


print('There are approx. {0:.2f} no. of session per day'.format(sessions_per_day['n_sessions'].median()))


# In[141]:


print('On an average a user has {0:.2f} no of session per day'.format(sessions_per_day['average_session_per_user'].mean()))


# In[142]:


#Split the distribution of sessions by device
sessions_per_day_by_device=data_visits.groupby(['start_ts_date','device']).agg({'uid':'count'})
sessions_per_day_by_device.columns = ['n_sessions']
sessions_per_day_by_device=sessions_per_day_by_device.reset_index()
sessions_per_day_by_device


# In[143]:


kwargs=dict(alpha=0.5,bins=30)
plt.hist(sessions_per_day_by_device[sessions_per_day_by_device['device']=='desktop']['n_sessions'],**kwargs,color='y',
          label='desktop')
plt.hist(sessions_per_day_by_device[sessions_per_day_by_device['device']=='touch']['n_sessions'],**kwargs,color='b',
          label='touch')
plt.gca().set(title='Session distribution on Device basis')
plt.ylabel('No. Of Session') 
plt.xlabel('Count') 
plt.legend()
plt.show()


# #### Conclusion
# 1. On an average each user has 1 session per day.
# 2. After grouping the data by device wise, we can say large portion of users still uses desktop than touch devices. which may be because the website is not that much user friendly for touch/mobile devices so we can ask the team to check the cause for this. If we are able to resolve the touch device issue surely we can further increase the count of session per day.

# ### - What is the length of each session?

# In[144]:


#Calculating session length
data_visits['session_length_sec']=(data_visits.end_ts-data_visits.start_ts).dt.seconds
data_visits.head()


# In[145]:


#plot histograme for session length.
plt.hist(data_visits['session_length_sec'],bins=100,range=[0,3000]) 
plt.title('Session Length')
plt.xlabel('Session Length')
plt.ylabel('No. of Sessions')
plt.show()


# As distribution is not normal so mean is not the correct value for average session length. So we will calculate median to find the average session length.

# In[146]:



avg_session_len=data_visits['session_length_sec'].mode()[0]
#print(avg_seesion_len) 


# In[147]:


print('Average Session length per day: {0} sec.'.format(avg_session_len)) 


# #### Conclusion:
# 
# So most of the users visit the app for 60 sec.

# ### - How often do users come back?

# In[148]:


first_activity_date = data_visits.groupby(['uid'])['start_ts_date'].min()
first_activity_date.name = 'first_activity_date'
data_visits = data_visits.join(first_activity_date,on='uid', how='left') 


# In[149]:


data_visits.shape


# In[150]:


data_visits['start_ts_date']=pd.to_datetime(data_visits['start_ts_date'])
data_visits['first_activity_date']=pd.to_datetime(data_visits['first_activity_date'])


# In[151]:


data_visits.info()


# In[152]:


data_visits['start_ts_monthly'] =  data_visits['start_ts_date'].astype('datetime64[M]') #pd.to_datetime(data_visits['start_ts_year'].astype(str)  + data_visits['start_ts_month'].astype(str), format='%Y%m')
data_visits['first_activity_month'] = data_visits['first_activity_date'].astype('datetime64[M]')#pd.to_datetime((data_visits['first_activity_date'].dt.year).astype(str)+(data_visits['first_activity_date'].dt.month).astype(str),  format='%Y%m')


# In[153]:


data_retention=data_visits
data_retention['cohort_lifetime_monthly'] = data_retention['start_ts_monthly'] - data_retention['first_activity_month']
data_retention['cohort_lifetime_monthly'] = data_retention['cohort_lifetime_monthly'] / np.timedelta64(1,'M')
data_retention['cohort_lifetime_monthly'] = data_retention['cohort_lifetime_monthly'].round().astype(int)
#data_visits['cohort_lifetime_monthly'] = data_visits['cohort_lifetime_monthly'].astype(int)


# In[154]:


data_retention['cohort_month']=data_retention['first_activity_month'].dt.strftime('%Y-%m') 
data_retention.head()


# In[155]:


cohorts_monthly = data_retention.groupby(['first_activity_month','cohort_lifetime_monthly']).agg({'uid':'nunique'}).reset_index() # Build the data frame with cohorts here


# In[156]:



cohorts_monthly.head()


# In[157]:


initial_users_count_monthly = cohorts_monthly[cohorts_monthly['cohort_lifetime_monthly'] == 0][['first_activity_month','uid']]


# In[158]:


initial_users_count_monthly.head()


# In[159]:


initial_users_count_monthly = initial_users_count_monthly.rename(columns={'uid':'cohort_users_monthly'})  # Rename the data frame column


# In[160]:


cohorts_monthly = cohorts_monthly.merge(initial_users_count_monthly,on='first_activity_month') # Join the data frames cohorts and initital_users_count


# In[161]:



cohorts_monthly.head()


# In[162]:



cohorts_monthly['retention'] = cohorts_monthly['uid']/cohorts_monthly['cohort_users_monthly']  # Calculate retention rate


# In[163]:



cohorts_monthly['first_activity_month'] = cohorts_monthly['first_activity_month'].dt.strftime('%Y-%m') 
cohorts_monthly=cohorts_monthly[cohorts_monthly.cohort_lifetime_monthly>0]
cohorts_monthly.head()
#data_retention_clean=data_retention[data_retention.cohort_lifetime_monthly>0]


# In[164]:



retention_pivot = cohorts_monthly.pivot_table(index='first_activity_month',columns='cohort_lifetime_monthly',values='retention',aggfunc='sum')


# In[165]:


display(retention_pivot)


# In[166]:


import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style='white')
plt.figure(figsize=(15, 5))
plt.title('Cohorts: User Retention')
sns.heatmap(
    retention_pivot, annot=True, fmt='.1%', linewidths=1, linecolor='gray'
) 


# ##### Conclusion

# 1. From the above graph, we can conclude that users retention rate is decreasing drastically for every cohort in every month.
# 2. June 2017 till November 2017 are doing better in 1 month as compared to other cohort. Among all september cohort retention rate is best in 1st month.
# 3. June cohort overall retention rate is better as compared to other cohort.

# ### Sales

# #### - When do people start buying? (In KPI analysis, we're usually interested in knowing the time that elapses between registration and conversion — when the user becomes a customer. For example, if registration and the first purchase occur on the same day, the user might fall into category Conversion 0d. If the first purchase happens the next day, it will be Conversion 1d. You can use any approach that lets you compare the conversions of different cohorts, so that you can determine which cohort, or marketing channel, is most effective.)

# In[167]:


data_visits.shape


# In[168]:


df_user_first_visit=data_visits.groupby('uid').agg({'start_ts':'min'}).reset_index()#['start_ts_date'].min()
df_user_first_visit.columns=['uid','first_visit']

df_user_first_order=data_orders.groupby('uid').agg({'buy_ts':'min'}).reset_index()
df_user_first_order.columns=['uid','first_order']


df_merged=df_user_first_visit.merge(df_user_first_order, on='uid', how='right')
df_merged.columns=['uid','first_visit','first_order']

df_merged.first_visit=df_merged.first_visit.fillna(df_merged.first_order)

df_merged.shape


# In[169]:


#checking if there is any records for whome order date is prior to visit date.
df_merged[df_merged.first_order<df_merged.first_visit]


# In[170]:


df_merged['conversion']=(df_merged.first_order-df_merged.first_visit).dt.days.astype(int)
#df_merged['conversion']=(df_merged.buy_ts-df_merged.first_visit).dt.days.astype(int)


# In[171]:


#display(df_merged.head())
#data_orders.shape


# In[172]:


data_conversion=pd.merge(data_orders[['uid','revenue','buy_ts']],df_merged, on='uid',how='left')
data_conversion['cohort_month']=data_conversion['first_order'].dt.strftime('%Y-%m') 
data_conversion.head()


# In[173]:


conversion_list=[0,1,2,7,14,30]
def conversion(conversion_day,group_by):
    grouped=(data_conversion.query('conversion<=@conversion_day').groupby(group_by)['uid'].nunique()
             /data_retention.groupby(group_by)['uid'].nunique())*100
    grouped=grouped.reset_index().rename(columns={'uid':'Conversion_{}%'.format(conversion_day)})
    return grouped


# In[174]:


#conversion(7,'cohort_month')


# In[175]:


new_conversion=[]
for i in conversion_list:
    new_conversion.append(conversion(i,'cohort_month'))


# In[176]:


#merging all columns and remove duplicates one.
new_conversion=pd.concat(new_conversion,axis=1)
new_conversion = new_conversion.loc[:,~new_conversion.columns.duplicated()]
new_conversion.head()


# In[177]:


#Code added v.1
#Distribution of conversion
data_conversion['conversion']
kwargs=dict(alpha=0.5,bins=30)
plt.hist(data_conversion['conversion'],**kwargs, label='Conversion')
plt.gca().set(title='Conversion')
plt.ylabel('Count') 
plt.xlabel('Conversion Time in days') 
plt.legend()
plt.show()


# #### Conclusion v.1
# Conversion% means how much % of users finally made purchase after their first visit. <br/>
# **Conversion_0%:**  means Conversion happens at 0 day. % of users made their first purchase at same day when they first visit the website. <br/>
# **Conversion_1%:** means Conversion happen at 1 day. % of users made their first purchase on 1st day after their first visit. <br/>
# If we see all retentions then we found that Retention is high for 0 day (Conversion_0%). and for rest of days it's increase with nominal %. like for the cohort '2017-06', 0 day conversion is 13.56% and 30 day conversion is 15.25% so in 30 days its increase by approx 2%. <br/> <br/>
# which means most of the users generally made purchase the same day on which they visit for the first time. This also depicts to us that may be website is selliing tickets for which users generally come to buy. 
# 

# In[178]:


data_conversion['conversion'].describe()


# #### Conclusion

# 1. Maximum conversion takes place in 0 days.
# 2. It seems from the conversion that the website sells something like tickets for which user generally come to buy something. 

# ### - How many orders do they make during a given period of time?

# In[179]:


data_orders.info()


# In[180]:


#data_visits[['uid','source_id','first_activity_date']].drop_duplicates()


# In[181]:


data_orders=data_orders.rename(columns=({'buy_ts':'order_datetime'}))
data_orders['order_year']=data_orders.order_datetime.dt.year
data_orders['order_week']=data_orders.order_datetime.dt.week
data_orders['order_date']=data_orders.order_datetime.dt.date


# In[182]:


first_order_date_by_customers = data_orders.groupby('uid')[ 'order_datetime'].min()
first_order_date_by_customers.name = 'first_order_date'
data_orders = data_orders.join(first_order_date_by_customers, on='uid')
data_orders['first_order_month'] = data_orders['first_order_date'].astype('datetime64[M]')
data_orders['order_month'] = data_orders['order_datetime'].astype('datetime64[M]')
print(data_orders.head(10)) 


# In[183]:



DAU_order = data_orders.groupby('order_date').agg({'uid': 'nunique','order_datetime':'count'})
WAU_order = data_orders.groupby(['order_year', 'order_week']).agg({'uid': 'nunique','order_datetime':'count'}).reset_index()
MAU_order = data_orders.groupby(['order_year', 'order_month']).agg({'uid': 'nunique','order_datetime':'count'}).reset_index()
DAU_order=DAU_order.rename(columns=({'uid':'n_users','order_datetime':'order_count'}))
WAU_order=WAU_order.rename(columns=({'uid':'n_users','order_datetime':'order_count'}))
MAU_order=MAU_order.rename(columns=({'uid':'n_users','order_datetime':'order_count'}))
#print(int(dau_total))
print(DAU_order)
print(WAU_order)
print(MAU_order) 


# In[184]:


#Finding Average order count year, month and weekly basis.
print('The Average daily order made: {0:.2f}'.format(DAU_order.order_count.mean()))
print('The Average weekly order made: {0:.2f}'.format( WAU_order.order_count.mean()))
print('The Average monthly order made: {0:.2f}'.format( MAU_order.order_count.mean()))


# ### What is the average purchase size?

# In[185]:


#data_orders.head(5)


# In[186]:


print('Average purchase size: $',round(data_orders.revenue.mean()))


# ### - How much money do they bring? (LTV)

# In[187]:


data_conversion['first_visit_month']=data_conversion['first_visit'].astype('datetime64[M]')
data_conversion['order_month']=data_conversion['buy_ts'].astype('datetime64[M]')
data_conversion.head()


# In[188]:


#comment out below code after changes recommended by reviewer v.1
''' 
data_conversion['cohort_age'] = (data_conversion['order_month'] - data_conversion['first_visit_month']) / np.timedelta64(1, 'M')
data_conversion['cohort_age'] = data_conversion['cohort_age'].round().astype('int')
data_conversion['cohort_month_ltv'] = data_conversion['first_visit_month'].dt.strftime('%Y-%m')
data_conversion
''' 


# In[189]:


#comment out below code after changes recommended by reviewer v.1
''' 
tot_ltv=pd.pivot_table(data_conversion,index=data_conversion['cohort_month_ltv'],columns='cohort_age', values='revenue', aggfunc='sum')
tot_ltv
''' 


# In[190]:


#comment out below code after changes recommended by reviewer v.1
''' 
tot_cum_ltv=pd.pivot_table(data_conversion,index=data_conversion['cohort_month_ltv'],columns='cohort_age', values='revenue', aggfunc='sum').cumsum(axis=1)
tot_cum_ltv
''' 


# In[191]:


#comment out below code after changes recommended by reviewer v.1
''' 
Avg_ltv=pd.pivot_table(data_conversion,index=data_conversion['cohort_month_ltv'],columns='cohort_age', values='revenue', aggfunc='mean')
Avg_ltv
''' 


# In[192]:


#comment out below code after changes recommended by reviewer v.1
''' 
plt.figure(figsize=(15, 8))
plt.title('Average customer lifetime value (LTV)')
sns.heatmap(
    Avg_ltv,
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='gray',
) 
'''


# In[193]:


#new code added v.1

data_conversion['first_order_month']=data_conversion['first_order'].astype('datetime64[M]')
cohort_sizes=data_conversion.groupby('first_order_month').agg({'uid':'nunique'})
data_tot_revenue= data_conversion.groupby(['first_order_month','order_month']).agg({'revenue':'sum'}).reset_index()
data_ltv=data_tot_revenue.merge(cohort_sizes,on='first_order_month')

data_ltv['ltv']=data_ltv['revenue']/data_ltv['uid']
data_ltv.head()


# In[194]:


data_ltv['cohort_lifetime'] = (data_ltv['order_month'] - data_ltv['first_order_month']) / np.timedelta64(1, 'M')
data_ltv['cohort_lifetime'] = data_ltv['cohort_lifetime'].round().astype('int')
data_ltv['cohort_month_ltv'] = data_ltv['first_order_month'].dt.strftime('%Y-%m')
data_ltv.head()


# In[195]:


output = data_ltv.pivot_table(
    index='cohort_month_ltv',
    columns='cohort_lifetime',
    values='ltv',
    aggfunc='mean')
output = output.cumsum(axis=1).round(2)
output


# In[196]:


plt.figure(figsize=(15, 8))
plt.title('Average customer lifetime value (LTV)')
sns.heatmap(
    output,
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='gray',
) 


# #### Conclusion
# 1. LTV for september and June cohorts are the best among all cohorts.
# 2. From 3rd month every cohort starts to pay except january and february.

# In[197]:


#output[[0, 1, 2, 3, 4, 5, 6]].cumsum(axis=1).mean(axis=0) 


# ## Marketing

# ### How much money was spent? Overall/per source/over time

# In[198]:


data_costs.info()


# In[199]:


data_costs


# In[200]:


#changing dt column datatype to datetime
data_costs['dt']=pd.to_datetime(data_costs['dt'], format="%Y-%m-%d")
data_costs=data_costs.rename(columns={'dt':'cost_date'})
data_costs


# In[201]:


data_costs['cost_month'] = data_costs['cost_date'].astype('datetime64[M]') 
data_costs


# In[202]:


#How much money was spent? Overall
print('Overall money spent : ',data_costs['costs'].sum())


# In[203]:


#How much money was spent? per source
data_source_cost=data_costs.groupby('source_id')['costs'].sum().reset_index()
data_source_cost.plot(kind="bar", title='Source wise cost',x='source_id', y='costs', figsize=(10,5))
plt.xlabel("Source")
plt.ylabel("Total Cost")


# Money was spent mostly on source_id=3

# In[204]:



#How much money was spent? over time
data_cost_month=data_costs.groupby('cost_month')['costs'].sum().reset_index()
data_cost_month
plt.figure(figsize=(10,5))
plt.plot(data_cost_month['cost_month'], data_cost_month['costs'],color='tab:red')
plt.title('Cost by month')
plt.xlabel('Year-Month')
plt.ylabel('Total Cost')
plt.show()


# #### Conclusion
# 1. Cost is highest for source_id=3
# 2. Cost starts to increase from september and reached to maximum in November and December. Later starts to decrease.

# ### How much did customer acquisition from each of the sources cost?

# In[205]:


#cost by source for each visitors
data_cost_bySource=data_costs.groupby('source_id').agg({'costs':'sum'}).reset_index()
data_visits_bySource=data_visits.groupby('source_id').agg({'uid':'nunique'})
data_cac=data_cost_bySource.merge(data_visits_bySource, on='source_id', how='left')
data_cac['cost_per_user']=data_cac['costs']/data_cac['uid']
data_cac


# In[206]:


#Bar chart for cost by source.
data_cac.plot(kind="bar", title='Source wise CAC-visitors',x='source_id', y='cost_per_user', figsize=(10,5))
plt.xlabel("Source")
plt.ylabel("Cost Per User")
plt.show()


# CAC of visitors is highest for source_id=3

# In[207]:


#Calculate unique source for each user from the visit table.

user_source_count=data_visits.groupby(['uid']).agg({'source_id':['count','nunique','min']}).reset_index()
user_source_count.columns=['uid','source_count','source_unique','source_id_single']
# Getting only those users who have single source
user_source_single=user_source_count[user_source_count.source_unique==1]


#Getting Users who have multiple sources and then find/update most common source for each user.
user_source_multi=user_source_count[user_source_count.source_unique>1].merge(data_visits[['uid','source_id']], on='uid', how='inner')
user_source_multi=user_source_multi.groupby(['uid'])['source_id'].agg(lambda x: x.mode()[0]).reset_index()

#remove all duplicate record.
user_source_single.rename(columns={'source_id_single':'source_id'},inplace=True)
user_source_single=user_source_single[['uid','source_id']]
user_source_single=user_source_single.drop_duplicates()
#remove all duplicate record.
user_source_multi=user_source_multi.drop_duplicates().reset_index()
#append tables.
user_source_final=user_source_single.append(user_source_multi)
user_source_final=user_source_final[['uid','source_id']].drop_duplicates().reset_index()
user_source_final


# In[208]:


#changes v.1 - new code added  
#cost by source for each depositor
data_cost_bySource=data_costs.groupby('source_id').agg({'costs':'sum'}).reset_index()
df_depositor_count=pd.merge(data_orders[['uid']],user_source_final, on='uid',how='inner')
data_depositer_bySource=df_depositor_count.groupby('source_id').agg({'uid':'nunique'})
data_cac_dep=data_cost_bySource.merge(data_depositer_bySource, on='source_id', how='left')
data_cac_dep['cost_per_user']=data_cac_dep['costs']/data_cac_dep['uid']
data_cac_dep


# In[209]:


#changes v.1 - code commented
#cost by source for each depositor
#data_cost_bySource=data_costs.groupby('source_id').agg({'costs':'sum'}).reset_index()
#data_depositer_bySource=user_source_final.groupby('source_id').agg({'uid':'nunique'})
#data_cac_dep=data_cost_bySource.merge(data_depositer_bySource, on='source_id', how='left')
#data_cac_dep['cost_per_user']=data_cac_dep['costs']/data_cac_dep['uid']
#data_cac_dep


# In[210]:


#Bar chart for cost by source.
data_cac_dep.plot(kind="bar", title='Source wise CAC-depositor',x='source_id', y='cost_per_user', figsize=(10,5))
plt.xlabel("Source")
plt.ylabel("Cost Per User")
plt.show()


# CAC for each depositer is high for source_id =3

# In[211]:


#changes v.1 - Existing code block commented
#cost by month
#data_cost_byMonth=data_costs.groupby('cost_month').agg({'costs':'sum'}).reset_index()
#data_visit_bymonth=data_visits.groupby('start_ts_monthly').agg({'uid':'nunique'}).reset_index().rename(columns={'start_ts_monthly':'cost_month'})
#data_cac_month=data_cost_byMonth.merge(data_visit_bymonth, on='cost_month', how='left')
#data_cac_month['cost_per_user']=data_cac_month['costs']/data_cac_month['uid']
#data_cac_month


# In[212]:


#changes v.1 - new code added  
#cost by month
data_cost_byMonth=data_costs.groupby('cost_month').agg({'costs':'sum'}).reset_index()
data_order_bymonth=data_orders.groupby('order_month').agg({'uid':'nunique'}).reset_index().rename(columns={'order_month':'cost_month'})
data_cac_month=data_cost_byMonth.merge(data_order_bymonth, on='cost_month', how='left')
data_cac_month['cost_per_user_per_month']=data_cac_month['costs']/data_cac_month['uid']
data_cac_month


# In[213]:


#changes v.1 - code block commented
#cost by source for each visitors
#data_cost_bySource = data_costs.groupby('source_id').agg({'costs':'sum'}).reset_index()
#source_id = data_visits.groupby('uid').agg({'source_id' : 'mean'}).reset_index()
#data_visits_bySource = data_orders.merge(source_id, on = "uid")

#data_visits_bySource


# In[214]:


plt.figure(figsize=(10,5))
plt.plot(data_cac_month['cost_month'],data_cac_month['cost_per_user_per_month'],color='tab:red')
plt.title('Cost Per User-Monthly')
plt.xlabel('Year-Month')
plt.ylabel('Cost Per User')
plt.show()


# Cost is highest in June, July and August.
# Cost suddenly increases in november, Janauary and April and then start to decrease.

# ### How worthwhile where the investments? (ROI)

# In[215]:


data_conversion=data_conversion.merge(user_source_final,on='uid',how='inner')
data_conversion_ltv=data_conversion.groupby(['source_id']).agg({'revenue':'sum', 'uid':'nunique'}).reset_index()
data_conversion_ltv.columns=['source_id','revenue','n_buyers']
data_conversion_ltv['ltv']= data_conversion_ltv['revenue'] / data_conversion_ltv['n_buyers']
data_romi=data_conversion_ltv.merge(data_cac_dep[['source_id','cost_per_user','costs']],on='source_id',how='outer')
data_romi['romi'] = data_romi['ltv'] / data_romi['cost_per_user']
data_romi['net_rev']=data_romi['revenue'] - data_romi['costs']
data_romi


# ## Conclusion
# 

# 1. After analysing the above metric I'll recomend the market department should invest in source_id 1 and 2 as rest of the sources's net revenue are not paying. They are in loss.
# 2. We should discard source 3 as it has the highest cost.
# 3. Cost of sources starts to increase from september and reached to maximum in November and December. Later starts to decrease.
# 3. As we can see the retention is very low and conversion is very high for 0 days. So with that we can say that the website sells something like tickets etc. for which users gerenrally come to purchase on the same day. 
# 4. After grouping the data by device wise, we can say large portion of users still uses desktop than touch devices. which may be because the website is not that much user friendly for touch/mobile devices so we can ask the team to check the cause for this. 
# 5. After looking at DAU and MAU, we can say that daily and monthly users visits are maximum in November and December and at at minimum in june, july, august and september.
# 6. Overall june, september and december cohort are better with respect LTV.
# 7. June cohort overall retention rate is better as compared to other cohort.
