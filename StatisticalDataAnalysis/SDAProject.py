#!/usr/bin/env python
# coding: utf-8

# # Analyzing plan for telecom operator Megaline
# 
# The company offers its clients two prepaid plans, Surf and Ultimate. The commercial department wants to know which of the plans brings in more revenue in order to adjust the advertising budget.
# 
# For analyzing, We have the data of 500 small Megaline client. The data includes who the clients are, where they're from, which plan they use, and the number of calls they made and text messages they sent in 2018. Our job is to analyze clients' behavior and determine which prepaid plan brings in more revenue.

# ## Step 1. Open the data file and study the general information. 

# In[4]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import seaborn as sns

#open files
df_calls=pd.read_csv('datasets/megaline_calls.csv')
df_internet=pd.read_csv('datasets/megaline_internet.csv')
df_messages=pd.read_csv('datasets/megaline_messages.csv')
df_plans=pd.read_csv('datasets/megaline_plans.csv')
df_users=pd.read_csv('datasets/megaline_users.csv')

#print general information of the dataframe
df_calls.info()
df_internet.info()
df_messages.info()
df_plans.info()
df_users.info()

#Print some data of each table.
display(df_calls.head())
display(df_internet.head())
display(df_messages.head())
display(df_plans.head())
display(df_users.head())


# ### Conclusion

# 1. By using info() method, we checked the general table information like columns name, their datatype and count of null values etc. 
# 2. All the calls data are loaded in df_calls table, similarly message data in df_messages , internet usage related data in df_internet, plan detail in df_plan and finally user detail in df_users. After excuting info() method, found that all the columns except chrun_date does not contain null values.
# 3. As shown above data is scatter in multiple tables so after analysing individual table we will first join the required data in one table and then analyze further.

# ## Step 2. Prepare the data

# Before merging all the tables, will identify if there is any invalid data in the individual tables.
# <br>
# We will check data for null or 0 values in  all the tables as there is no point to include 0 calls or gb used etc. as it will brings no or mininal revenue to the company. 

# In[5]:


#checking for null or 0 values in Call_duration table.
print('###############################Call Duration Table#######################################################')
print('Count of null values data',len(df_calls[(df_calls['duration'].isnull())]))
print('Count of 0 values data',len(df_calls[(df_calls['duration']==0)]))
#delete all rows for 0 value data.
df_calls=df_calls[(df_calls['duration']!=0)]
print('After deleting count of data', len(df_calls))


# In[6]:


#checking for null or 0 values in Internet table.
print('###############################Internet Table#######################################################')
print('Count of null values data',len(df_internet[(df_internet['mb_used'].isnull())]))
print('Count of null values data',len(df_internet[(df_internet['mb_used']==0)]))
#delete all 0 values data.
df_internet=df_internet[(df_internet['mb_used']!=0)]
print('After deleting count of data', len(df_internet))


# In[7]:


#checking for null or 0 values in Message table.
print('###############################Message Table#######################################################')
print(len(df_messages[(df_messages['message_date'].isnull())]))
print(len(df_messages[(df_messages['message_date']==0)]))


# Now check for users table.

# In[8]:


#checking for null or 0 values in Users table.
print('###############################Users Table#######################################################')
display(df_users[df_users['user_id'].isnull()])
display(df_users[df_users['first_name'].isnull()])
display(df_users[df_users['last_name'].isnull()])
display(df_users[df_users['age'].isnull()])
display(df_users[df_users['city'].isnull()])
display(df_users[df_users['reg_date'].isnull()])
display(df_users[df_users['plan'].isnull()])
display(df_users[df_users['churn_date'].isnull()])


# Found null values only in churn_date column but this is already mention in the requirement that if the value is missing, the calling plan was being used when this data was retrieved so it will stay as it is.
# <br>
# <br>
# Now we will also verify other columns for any invalid data.

# In[9]:


print('Min. and Max. age are: {0} and {1}'.format(df_users['age'].min(), df_users['age'].max()))


# In[10]:


print('Min. and Max. reg_date are: {0} and {1}'.format(df_users['reg_date'].min(), df_users['reg_date'].max()))


# In[11]:


df_users['plan'].value_counts()


# We have to also delete the data from calls, Internet and Message table where records in these table exists before the
# reg_date and after the churn_date w.r.t to each user. As these records are considered to be invalid because before registration user can't use the service and same after the deregistration.

# In[12]:


#display(df_users)


# In[13]:


print('Total count before removing invalid calls data: ', len(df_calls))
print('Total count before removing invalid messages data: ', len(df_messages))
print('Total count before removing invalid gb_used data: ', len(df_internet))
for index, row in df_users.iterrows():
    user_id = row['user_id']
    churn_date = row['churn_date']
    reg_date = row['reg_date']
    invalid_calls=df_calls[(df_calls['user_id']==user_id) & ((df_calls['call_date']<reg_date)|(
        df_calls['call_date']>(churn_date if churn_date else df_calls['call_date'])))  
             ]
    invalid_messages=df_messages[(df_messages['user_id']==user_id) & ((df_messages['message_date']<reg_date)|(
        df_messages['message_date']>(churn_date if churn_date else df_messages['message_date'])))  
             ] 
    invalid_gb_used=df_internet[(df_internet['user_id']==user_id) & ((df_internet['session_date']<reg_date)|(
        df_internet['session_date']>(churn_date if churn_date else df_internet['session_date'])))  
             ] 
    
    if (len(invalid_calls)>0):
        #print(str(user_id)+'--'+str(churn_date)+'---'+str(reg_date))
        df_calls.drop(invalid_calls.index,inplace=True)
    if (len(invalid_messages)>0):
        df_messages.drop(invalid_messages.index,inplace=True)
    if (len(invalid_gb_used)>0):
        df_internet.drop(invalid_gb_used.index,inplace=True)

print('Total count after removing invalid calls data: ', len(df_calls))
print('Total count after removing invalid messages data: ', len(df_messages))
print('Total count after removing invalid gb_used data: ', len(df_internet))


# We have removed invalid data from calls, messages and internet table. We calculated the invalid data on basis of chur_date. One think I have noticed here that there are lots of invalid data available in these tables for which duration is also greater than 0 so we need to check with the team how users are able to call after deregistration and who will pay for that?

# So far we deleted all invalid data found in the tables. 
# <br>
# Now we will also incorporate some conditions given so that we can easily make our analysis as per the project requirement.
# 1. First all individual call duration should be round up to minutes.
# 2. Also internet usage should be round up from mb to gb. It should be done considering the total for the month not the individual web sessions.
# 3. For all these tables we also need month column to calculate total revenue monthly.
# 
# 

# In[14]:


df_calls['month']=pd.DatetimeIndex(df_calls['call_date']).month#adding month column in the calls table.
df_calls['duration']=df_calls['duration'].apply(np.ceil).astype(int)# all call's seconds are rounded up to minute.

df_internet['month']=pd.DatetimeIndex(df_internet['session_date']).month#adding month column in the internet table.
df_messages['month']=pd.DatetimeIndex(df_messages['message_date']).month#adding month column in the message table.


# **Now grouping the data by user_id and month so that we can calculate total call_dutaion, messages and mb used for each month by each user.**

# In[15]:


#Create grouped data by users and months.
df_calls_group=df_calls.groupby(['user_id','month']).agg({'duration':['sum'], 'call_date':'count'})
df_internet_group=df_internet.groupby(['user_id','month']).agg({'mb_used':['sum']})
df_messages_group=df_messages.groupby(['user_id','month']).agg({'message_date':['count']})

df_calls_group.columns=['Call_duration','Call_count']
df_internet_group.columns=['mb_used']
df_messages_group.columns=['message_count']

# megabytes rounded up to gigabytes
df_internet_group['gb_used']=df_internet_group['mb_used'].apply(lambda x: math.ceil(x/1024))

display(df_calls_group.head())
display(df_internet_group.head())
display(df_messages_group.head())
print('Total count of calls, internet and messages table are: ',str(len(df_calls_group))+','+str(len(df_internet_group))+','+str(len(df_messages_group)))


# **Now merge all tables.**
# <br/>

# In[16]:


#merging
df_temp= pd.merge(left = df_calls_group , 
                right = df_internet_group, how='outer',on=['user_id', 'month']).fillna(0)
df_merged=pd.merge(left = df_temp , 
                right = df_messages_group, how='outer',on=['user_id', 'month']).fillna(0)


# In[17]:


print('Total of rows count after merging: ',len(df_merged))
#print(df_merged.sort_values(by=['user_id','month']))
df_merged=df_merged.sort_values(by=['user_id','month']).reset_index()
display(df_merged.head())


# In[18]:


df_merged_plan= pd.merge(left = df_merged , 
               right = df_users[['user_id','plan']], how='outer',on=['user_id']).fillna(0)
display(df_merged_plan.head())
print('Total of rows count after merging with plan table: ',len(df_merged_plan))


# Checking basic information of new merged table.

# In[19]:


df_merged_plan.info()
print('Total records with 0 call_duration: ',len(df_merged_plan[df_merged_plan['Call_duration']==0]))
print('Total records with 0 Call_count: ',len(df_merged_plan[df_merged_plan['Call_count']==0]))
print('Total records with 0 mb_used: ',len(df_merged_plan[df_merged_plan['mb_used']==0]))
print('Total records with 0 message_count: ',len(df_merged_plan[df_merged_plan['message_count']==0]))

#print()


# **Convert the data to the necessary types in the final table**

# In[20]:


#Change the required data type.
df_merged_plan['month'] = df_merged_plan['month'].astype('int8')
#df_merged_plan['Call_duration'] = df_merged_plan['Call_duration'].astype('float')
df_merged_plan['Call_count'] = df_merged_plan['Call_count'].astype('int64')
df_merged_plan['message_count'] = df_merged_plan['message_count'].astype('int32')
df_merged_plan['gb_used'] = df_merged_plan['gb_used'].astype('int32')


# In[21]:


df_merged_plan.info()


# In[22]:


df_merged_plan


# Now let's calculate total_revenue for each month. For this I first write function to calculate all monthly cost based on below logic.
# <br>
# <br>
# **Surf**
# 1. Monthly charge: \\$20
# 2. 500 monthly minutes, 50 texts, and 15 GB of data
# 3. After exceeding the package limits:
#     1. 1 minute: 3 cents
#     2. 1 text message: 3 cents
#     3. 1 GB of data: \\$10

# **Ultimate**
# 1. Monthly charge: \\$70
# 2. 3000 monthly minutes, 1000 text messages, and 30 GB of data
# 3. After exceeding the package limits:
#     1. 1 minute: 1 cent
#     2. 1 text message: 1 cent
#     3. 1 GB of data: \\$7

# **Total_Monthly_Charges=Monthly charge+Charges for extra min.+Charges for extra message+Charges for extra GB.**

# In[23]:


def monthly_cost(row):
   
    plan=row['plan']#plan type
    #getting actual used call_duration, message and gb used from the row and save in the variable.
    actual_Call_duration=row['Call_duration']
    actual_Call_count=row['Call_count']
    actual_gb_used=row['gb_used']
    actual_message_count=row['message_count']
    
    #getting plan detail from the table and store all the required field in to the variable.
    df_row_plan=df_plans[df_plans['plan_name']==plan]
    minutes_included=int(df_row_plan['minutes_included']) 
    messages_included=int(df_row_plan['messages_included'])
    mb_per_month_included=int(df_row_plan['mb_per_month_included'])
    gb_per_month_included=int(mb_per_month_included/1024)
    
    usd_monthly_pay=float(df_row_plan['usd_monthly_pay'])
    usd_per_gb=float(df_row_plan['usd_per_gb'])
    usd_per_message=float(df_row_plan['usd_per_message'])
    usd_per_minute=float(df_row_plan['usd_per_minute'])
   
    
    extra_call_charges=0
    extra_mess_charges=0
    extar_gb_charges=0
    try:

        if(actual_Call_duration>minutes_included):
            diff_call_duration=actual_Call_duration-minutes_included
            extra_call_charges=diff_call_duration*usd_per_minute
        else:
            extra_call_charges=0

        if(actual_message_count>messages_included):
            diff_msg=actual_message_count-messages_included
            extra_mess_charges=diff_msg*usd_per_message
        else:
            extra_mess_charges=0

        if(actual_gb_used>gb_per_month_included):
            diff_gb=actual_gb_used-gb_per_month_included
            extar_gb_charges=diff_gb*usd_per_gb
        else:
            extar_gb_charges=0
        #total_revenue=usd_monthly_pay+extra_call_charges+extra_mess_charges+extar_gb_charges
    except:
          print("An exception occurred")
        
        
    total_revenue=(usd_monthly_pay)+(extra_call_charges)+(extra_mess_charges)+(extar_gb_charges)
    return total_revenue
    
    
    
#df_merged_plan['total_revenue']=df_merged_plan.apply(monthly_cost,axis=1)
#print(df_merged_plan.head())


# In[24]:


df_merged_plan['total_monthly_revenue']=df_merged_plan.apply(monthly_cost,axis=1)
print(df_merged_plan.tail())


# ### Conclusion
# So far, we have deleted the invalid data, changed the required data type and added the required columns. Also total_monthly_revenue has been calculated and saved in the final table with all other required columns. So in the next step we will use our this final table make the analysis.

# ## Step 3. Analyze the data

# In[25]:


print(df_merged_plan.head(10))


# In[26]:


df_merged_plan.info()
df_merged_plan.describe()


# Below section of code will find the Avg. minutes, texts, and volume of data the users of each plan require per month.

# In[27]:



# Find the minutes, texts, and volume of data the users of each plan require per month.
df_surf=df_merged_plan[df_merged_plan.plan=='surf']
df_ultimate=df_merged_plan[df_merged_plan.plan=='ultimate']
print('Surf users detail: call duration mean:{0:.2f}, Variance:{1:.2f}, Std. dev.{2:.2f}: '.format(df_surf['Call_duration'].mean(),
                                                                                      df_surf['Call_duration'].var(),
                                                                                      df_surf['Call_duration'].std()))
print('Surf users detail: Messages mean:{0:.2f}, Variance:{1:.2f}, Std. dev.{2:.2f}: '.format(df_surf['message_count'].mean(),
                                                                                      df_surf['message_count'].var(),
                                                                                      df_surf['message_count'].std()))
print('Surf users detail: Mb_used mean:{0:.2f}, Variance:{1:.2f}, Std. dev.{2:.2f}: '.format(df_surf['mb_used'].mean(),
                                                                                      df_surf['mb_used'].var(),
                                                                                      df_surf['mb_used'].std()))

print('Ultimate users detail: call duration mean:{0:.2f}, Variance:{1:.2f}, Std. dev.{2:.2f}: '.format(df_ultimate['Call_duration'].mean(),
                                                                                      df_ultimate['Call_duration'].var(),
                                                                                      df_ultimate['Call_duration'].std()))


print('Ultimate users detail: Messages mean:{0:.2f}, Variance:{1:.2f}, Std. dev.{2:.2f}: '.format(df_ultimate['message_count'].mean(),
                                                                                      df_ultimate['message_count'].var(),
                                                                                      df_ultimate['message_count'].std()))


print('Ultimate users detail: Mb_used mean:{0:.2f}, Variance:{1:.2f}, Std. dev.{2:.2f}: '.format(df_ultimate['mb_used'].mean(),
                                                                                      df_ultimate['mb_used'].var(),
                                                                                      df_ultimate['mb_used'].std()))


# ### Histogram for the parameters: Call_duration, messages, gb_used

# In[28]:


hist_list=['Call_duration', 'message_count', 'gb_used']
for param in hist_list:
    df_ultimate=df_merged_plan.loc[df_merged_plan.plan=='ultimate', param]
    df_surf=df_merged_plan.loc[df_merged_plan.plan=='surf', param]
    ##if(param =='Call_duration'):
     #   kwargs=dict(alpha=0.6,bins=100,density=True,stacked=True)
     #   ylabel='Frequency Density'
    #else:
    kwargs=dict(alpha=0.5,bins=100)
    ylabel='No. of Users'
    plt.hist(df_ultimate,**kwargs,color='y', label='Ultimate')
    plt.hist(df_surf,**kwargs,color='b', label='Surf')
    plt.gca().set(title='Ultimate vs. Surf: Displaying '+param+' by Users')
    plt.xlabel(param) 
    plt.ylabel(ylabel) 
    plt.legend()
    plt.show()
    


# As we can see from the above graph, Surf users are dominating for every parameters but also there seems to have outliers so first we try to remove all those and then comapre the Histogrma again.

# #### Conclusion: Comparison of Surf and Ultimate users:
# 
# **Call_duration:**
# - As we can see from the graph above, In General most of the users uses calls duration between 250 and 620 minutes. In which  Surf users uses mostly between 300 to 600 min. while Ultimate users uses mostly 400 min.
# - Very few users uses time more than 1000 min.
# - There are more users of Surf plan.
# <br><br>
# **Messages:**
# - Many of users does not sent messages at all.
# - Surf users mostly sent messages between 1 to 50. while the most of the ultimate users sent berween 1 to 70. This is may be due to the condition that surf users got free messages till 50.
# - There are more users of Surf plan.
# <br><br>
# **Gb_used**
# - Most of the surf users uses internet between 15-25 gb. While the range is almost same for ultimate users as well.
# - Very few users use internet more the 30 gb.
# <br><br>
# As we can see in the grapg there are outliers as well in the data so now we will try to remove the outliers and then see the graph again.
# 

# ### Removing outliers.

# Removing rows where all three parameters are 0. As I think there may be the customer who are very new and they have not even started to use the plan. So they wont be usefull to us to analyze the customer behaviours.

# In[29]:


#pd.set_option('display.max_rows', None)
print('Data count before removing: ',len(df_merged_plan))
#display(len(df_merged_plan[df_merged_plan['Call_duration']==0]))
df_merged_plan_fil=df_merged_plan.loc[~((df_merged_plan['Call_duration']==0) & (df_merged_plan['message_count']==0) 
                           &(df_merged_plan['gb_used']==0))]
print('Data count after removing 0 calls_duration, messages and gb_used. ',len(df_merged_plan_fil))


# In[30]:


df_merged_plan_fil


# In[31]:


#Viewing outliers.
import seaborn as sns

ax=sns.boxplot(x='plan',y=df_merged_plan_fil['Call_duration'],data=df_merged_plan_fil)
plt.show()
ax=sns.boxplot(x='plan',y=df_merged_plan_fil['message_count'],data=df_merged_plan_fil )
plt.show()
ax=sns.boxplot(x='plan',y=df_merged_plan_fil['gb_used'] ,data=df_merged_plan_fil)
plt.show()


# ### Conclusion: Looking for outliers:
# 1. Call_Duration: For both the plans there are some users who are calling more than 1000 min.(upper limit is 1000 min.). So the outlier range for surf plan is 1000 to 1500 min. and for ultimate plan is 1000 to 1400 min. 
# 
# 2. Message: Few outliers are there in ultimate as compare to surf users. 25% of all surf users sent more than 125 messages. and one of them send 275 message. So surf users has outlier range between 125 till 275 messages while for ultimate users outliers range is between 140 till 170. Also there are users who send 0 messages for surf plan.
# 
# 3. Gb_used: For surf plan there are outliers range starts from approx.32 to 70 gb and for ultimate approx.31 to 46 gb.
# 
# In General, I can say that there are many outliers available in the data so we need to remove them before analyzing. 

# As we can see above there are outliers available in each parameters. So using Z-Score we try to remove those outliers.

# In[32]:


#Calculate zscore for the parameters.
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_merged_plan_fil[['Call_duration','message_count','gb_used']]))
#print(z)
threshold = 3
print(np.where(z > 3))


# Below code will removing outliers.

# In[33]:


df_merged_plan_fil_o = df_merged_plan_fil[(z < 3).all(axis=1)]
print(df_merged_plan_fil.shape)
print(df_merged_plan_fil_o.shape)


# In[34]:


#Viewing outliers after deleting outliers.
import seaborn as sns
ax=sns.boxplot(x='plan',y=df_merged_plan_fil_o['Call_duration'],data=df_merged_plan_fil)
plt.show()
ax=sns.boxplot(x='plan',y=df_merged_plan_fil_o['message_count'] ,data=df_merged_plan_fil)
plt.show()
ax=sns.boxplot(x='plan',y=df_merged_plan_fil_o['gb_used'] ,data=df_merged_plan_fil)
plt.show()


# ### Histogram for the parameters (Call_duration, messages, gb_used) after removing outliers. 

# In[35]:


hist_list=['Call_duration', 'message_count', 'gb_used']
for param in hist_list:
    df_ultimate=df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan=='ultimate', param]
    df_surf=df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan=='surf', param]
    ##if(param =='Call_duration'):
     #   kwargs=dict(alpha=0.6,bins=100,density=True,stacked=True)
     #   ylabel='Frequency Density'
    #else:
    kwargs=dict(alpha=0.5,bins=100)
    ylabel='No. of Users'
    plt.hist(df_ultimate,**kwargs,color='y', label='Ultimate')
    plt.hist(df_surf,**kwargs,color='b', label='Surf')
    plt.gca().set(title='Ultimate vs. Surf: Displaying '+param+' by Users')
    plt.xlabel(param) 
    plt.ylabel(ylabel) 
    plt.legend()
    plt.show()
    


# #### Conclusion: Comparison of Surf and Ultimate users after removal of outliers:
# 
# **Call_duration:**
# - After removal of outliers, Max. range has been shorten. Its. has been set as approx. 1200. min.
# - rest the analyses remains same as we found before the removal of outliers.
# <br><br>
# **Messages:**
# - Data is not distributed normally. it is skewed to the right.
# - Max range has been set to 120+. So now we can also say most of the users sent 40 to 50 messages in a month.
# <br><br>
# **Gb_used**
# - This is somewhat normally distributed graph.
# - Most of the users use between 14 to 21 gb.
# - Max. range is now set to 40gb.
#  

# In[36]:


#hist_list=['Call_duration', 'message_count', 'gb_used']
#for param in hist_list:
 #   Q1=df_merged_plan_fil[param].quantile(0.25)
  #  Q3=df_merged_plan_fil[param].quantile(0.75)
   # IQR=Q3-Q1
    #quer=param+ '>=(@Q1-1.5*@IQR) & '+param+'<=(@Q3+1.5*@IQR)'
    #print ()
    #data_raw_fil=df_merged_plan_fil.query(quer)
    #print('Data count after removing outliers for '+ param+':' ,len(data_raw_fil))
    
df_merged_plan_fil_o.info()
print('###########################################################################################################')
print()
print('Surf Users Detail')
display(df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan=='surf'].describe())
print('###########################################################################################################')
print('Ultimate Users Detail')
display(df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan=='ultimate'].describe())


# ### Conclusion:
# **The Mean of Surf and Ultimate users are (approx):**.
# - Call_Duration: The Mean values of both plan are 416 and 415 which is approx same. So we can say on an average users of both plan are using 415 min. per month. Also if we compare the mean with median we found median is more than mean for ultimate and almost equal to surf users. That's mean  there are more users of ultimate plan who spending less minutes. 
# - Messages: The Mean values of both the plan are 28 and 35. thats conclude that the users of surf plan send less messages on avg. as compare to ultimate plan. Also median values are 23 and 29. For both the plan Median are less than mean that's mean more data available for those users who send more messages.
# - Gb_used: Mean of both plan are almost same. So plan type does not effect the internet usage.<br><br>
# **The Standard Deviation of Surf and Ultimate users are (approx):**.
# - Call_Duration: Approx 223 and 222. Std. Dev are almost equal for both plan. This also revels that there is high deviation in calls duration w.r.t to mean values of both users.
# - Messages: 28 and 31. The std. dev. is high. It revels that how much each data differs from the mean of messages.
# - Gb_used: Approx. 7 for both plan. The deviation is equal for both users. Now each data is deviated by 7 gb w.r.t mean for each type users.
# 
# 

# ### Conclusion

# 1. <b>Comparison of Call_duration between Ultimate and Surf Users:</b> Now after removing outliers, Graph looks somewhat normally distributed except the records which are there for 0 call_duration. Also we can see thet Surf users are dominating in comaprison to ultimate users for call_duration.  
# 2. <b>Comparison of message_count between Ultimate and Surf Users:</b> This graph shows that data is skewed to the right. And also Surf users again dominate the Ultimate Users.
# 3. <b>Comparison of gb_used between Ultimate and Surf Users:</b> Data is normally distributed. And as in all above case Surf Users are using the internet more than Ultimate Users.
# 

# ## Step 4. Test the hypotheses

# ### The average revenue from users of Ultimate and Surf calling plans differs.

# In[37]:


#Test the hypotheses.
from scipy import stats as st
import numpy as np

surf_user_monthly_revenue=df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan=='surf']['total_monthly_revenue']
ultimate_user_monthly_revenue=df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan=='ultimate']['total_monthly_revenue']
print('Variance of Surf Users ', surf_user_monthly_revenue.var())
print('Variance of Ultimate Users ',ultimate_user_monthly_revenue.var())

#print(surf_user_monthly_revenue.head())
#print(ultimate_user_monthly_revenue.head())

alpha = .05 # critical statistical significance level
results = st.ttest_ind(
        surf_user_monthly_revenue, 
        ultimate_user_monthly_revenue,equal_var=False)#We pas equal_var as False as the variance of both sample are not equal.
print('p-value: ', results.pvalue)

if (results.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis") 

print('Avg. revenue from Surf Users: {0}'.format(surf_user_monthly_revenue.mean()))   
print('Avg. revenue from Ultimate Users: {0}'.format(ultimate_user_monthly_revenue.mean()))


# ### Conclusion
# As we have two statistical populations here which are based on same samples so we apply the method scipy.stats.ttest_ind(array1, array2, equal_var).<br/>
# In this scenario, We have the following hypothesis:<br/>
# **Null Hypothesis H0:** The average revenue from users of Ultimate and Surf calling plans does not differs.
# <br/>
# **Alternative Hypothesis H1:** The average revenue from users of Ultimate and Surf calling plans differs.
# <br/>
# After examine the p-value, We can say that we have to reject the null hypothesis, which implies average revenue from users of Ultimate and Surf calling plans differs.

# ### The average revenue from users in NY-NJ area is different from that of the users from other regions.

# In[38]:


#Test the hypotheses.
from scipy import stats as st
import numpy as np
df_users['city']=df_users['city'].str.lower()
df_users['city'].value_counts()
df_ny_nj_users=df_users[df_users['city'].str.contains('|'.join(['ny-nj','ny','nj']))]
df_ny_nj_users_merg= df_merged_plan_fil_o.loc[(df_merged_plan_fil_o.user_id.isin(df_ny_nj_users['user_id']))]
df_except_nynj_users_merg=df_merged_plan_fil_o.loc[~(df_merged_plan_fil_o.user_id.isin(df_ny_nj_users['user_id']))]

print('Variance of Surf Users ', df_ny_nj_users_merg['total_monthly_revenue'].var())
print('Variance of Ultimate Users ',df_except_nynj_users_merg['total_monthly_revenue'].var())
alpha = .05 # critical statistical significance level
results = st.ttest_ind(
        df_ny_nj_users_merg['total_monthly_revenue'], 
        df_except_nynj_users_merg['total_monthly_revenue'],equal_var=False)#We pas equal_var as False as the variance of both sample are not equal.
print('p-value: ', results.pvalue)

if (results.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis") 

print('Avg. revenue from users in NY-NJ area: {0}'.format(df_ny_nj_users_merg['total_monthly_revenue'].mean()))   
print('Avg. revenue from users not in NY-NJ area: {0}'.format(df_except_nynj_users_merg['total_monthly_revenue'].mean()))


# ### Conclusion
# As we have two statistical populations here which are based on same samples so we apply the method scipy.stats.ttest_ind(array1, array2, equal_var).<br/>
# In this scenario, We have the following hypothesis:<br/>
# **Null Hypothesis H0:** The average revenue from users in NY-NJ area is not different from that of the users from other regions.
# <br/>
# **Alternative Hypothesis H1:** The average revenue from users in NY-NJ area is different from that of the users from other regions.
# <br/>
# After examine the p-value, We can say that we can't reject the null hypothesis, which implies that average revenue from users in NY-NJ area is not different from that of the users from other regions.
# We also have calculated the avg. revenues of both the regions and found no such difference.

# ## Step 5. Write an overall conclusion

# In[39]:


#df_merged_plan_fil_o.loc[df_merged_plan_fil_o.plan='Surf', ]
df_merged_plan_fil_o_pvt=df_merged_plan_fil_o.pivot_table(
    index='month', columns='plan', values='total_monthly_revenue', aggfunc='mean').reset_index()

#df_merged_plan_fil_o_pvt.columns=['month','Total_Revenue_Surf','Total_Revenue_Ultimate']
print(df_merged_plan_fil_o_pvt)


# In[40]:


df_merged_plan_fil_o_pvt.plot(kind="bar", title='Surf vs Ultimate: Total Monthly Revenue',x='month')
plt.xlabel("Month")
plt.ylabel("Total Monthly Revenue")


# **As we can see from the above graph that average monthly revenue is more for ultimate users**.
# So now I can conclude that ultimate plan bring more revenues so will recommend to our commercial department to adjust the advertising budget on this plan more. 
# Also during analyzing we found some data which needs further investigation like: some records of calls, messages and gb_used which were recorded after deregistration by users. we store them in seprate table and will share with the team to find the cause for that.

# ### Conclusion:
# 

# 1. If we see the average monthly revenue, We find avg. revenue from Surf plan is approx.57 and for Ultimate is approx.72. so it shows that on an average monthly revenues are more for ultimate users.
# So now I can conclude that Ultimate plan bring more revenues so will recommend to our commercial department to adjust the advertising budget on ultimate plan more. 
# 
# 2. Also during analyzing we found some data which needs further investigation like: some records of calls, messages and gb_used which were recorded after deregistration by users. we store them in seprate table and will share with the team to find the cause for that.

# ### Conclusion
# 1. Data was scatter in multiple tables, so we first checked inidvidual table for any invalid data and then merged all tables in single.
# 2. We removed invalid data from calls, messages and internet table We calculated the invalid data on basis of chur_date. One think I have noticed here that there are lots of invalid data available in these tables for which duration is greater than 0 (i.e. the calls, message and gb_used date is greater than the churn date) so we need to check with the team how users are able to call after deregistration and who will pay for that?
# 3. After merging, we also added some column for analyzing purpose. like total_monthly_revenue.
# 4. As our analysis should be focused on surf and ultimate user's behaviour so we separate all the data in two types of plan and then check all the three parameters(call_duration, message count and gb_used) for both types of users.
# 5. After plotting the box plot we identified that there are some outiers available in the data so first I try to delete them and then made analysis.
# 6.  After deleting the outlier: I calculated Mean, Variance and std. dev. of all three parameters for both users. The mean value of call_duration of both the plan are almost same i.e. 416 and 415. About messages I can that users are not much interested in using messages service as most of them are not even using all the free messages specially for ultimate users. Mean of gb_used is almost same for both type of users i.e approx 17. 
# 7. In the section 'Test the hypotheses', we got proved that avg. revenue from the users of ultimpate and surf plan are differs. 
# 8. we also plotted the bar graph of avg. monthly reveue of both the users, which concluded that average monthly revenue is more for ultimate users than surf users. <br>
# **The avg. revenue from Surf plan is approx.57 and for Ultimate is approx.72.**<br>
# **So now I can conclude that Ultimate plan bring more revenues so will recommend to our commercial department to adjust the advertising budget on ultimate plan more.**

# In[ ]:




