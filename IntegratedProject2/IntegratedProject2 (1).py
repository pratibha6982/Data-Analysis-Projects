#!/usr/bin/env python
# coding: utf-8

# # Project Description:

# I work at a startup that sells food products. I need to investigate user behavior for the company's app.
# First we need to study the sales funnel. Find out how users reach the purchase stage. How many users actually make it to this stage? How many get stuck at previous stages? Which stages in particular? and then look at the results of an A/A/B test.<br/>
# <br/>
# Our Goal is to investigate user behaviour. How they are behaving in each stage of the product. Also our designers would like to change the fonts for the entire app but managers are afraid the users might find the new design intimidating. So to make decision whether we should incorporate the new changes, we need to perform A/A/B test. Based on the result our manager will decide for the implementation of new changes.
# <br/>
# The project is divided into 5 task. Please use below links to go particular section promptly.
# 1. <a href='#step1'>Step 1. Open the data file and read the general information</a>
# 2. <a href='#step2'>Step 2. Prepare the data for analysis</a>
# 3. <a href='#step3'>Step 3. Study and check the data</a>
# 4. <a href='#step4'>Step 4. Study the event funnel</a>
# 5. <a href='#step5'>Step 5. Study the results of the experiment</a>

# <a id='step1'></a>
# ## Step 1. Open the data file and read the general information

# In[4]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install plotly -U')
get_ipython().system('pip install --upgrade -q seaborn')


# In[2]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime as dt
import warnings

import seaborn as sns
import plotly.express as px 
from plotly import graph_objects as go

import math
from scipy import stats

warnings.filterwarnings('ignore')
#pd.set_option("display.max_rows", 10)
pd.set_option('max_colwidth', 400)


# In[3]:


#Url Path for the data.
Data_foodLogs_url='logs_exp_us.csv'

#Loading files locally in to dataframe
data_foodLogs=pd.read_csv(Data_foodLogs_url,sep='\\t')

#Verifying files and data.
display(data_foodLogs.head())
display(data_foodLogs.tail())
display(data_foodLogs.sample())


# <a id='step2'></a>
# ## Step 2. Prepare the data for analysis

# In[4]:


data_foodLogs.info(memory_usage='deep')


# In[5]:


data_foodLogs.rename(columns={'EventName':'event_name','DeviceIDHash':'user_id','EventTimestamp':'event_datetime','ExpId':'exp_id'
                             },inplace=True)
data_foodLogs


# In[6]:


data_foodLogs.isnull().sum()


# In[7]:


print('Count of duplicate records: ',data_foodLogs.duplicated().sum())
duplicateData=data_foodLogs[data_foodLogs.duplicated()]
print('Percentage of duplicate records are: {0}%'.format(round(len(duplicateData)/len(data_foodLogs)*100,2)))


# In[8]:


#drop all duplicate data.
data_foodLogs.drop_duplicates(inplace=True)


# In[9]:


data_foodLogs.event_name.unique()


# In[10]:


data_foodLogs[data_foodLogs.user_id<1000000000000]


# In[11]:


print(data_foodLogs.event_datetime.min())
print(data_foodLogs.event_datetime.max())


# In[12]:


data_foodLogs.exp_id.unique()


# In[13]:


data_foodLogs['event_datetime']=pd.to_datetime(data_foodLogs['event_datetime'], unit='s')
data_foodLogs


# In[14]:


data_foodLogs.info()


# In[15]:


data_foodLogs['event_date']=data_foodLogs['event_datetime'].dt.date
data_foodLogs['event_time']=data_foodLogs['event_datetime'].dt.time
data_foodLogs


# #### Conclusion:
# In the Data Preprocessing task, first I rename the all columns then check for any null or duplicates values. I found around 413 duplicated value which I dropped otherwise it will impact the analysis. After this, I verify data in all the columns by using unique , min or max functions. Also changed the EventTimestamp column data in the readable date format. And at last added two different columns for date and time.

# <a id='step3'></a>
# ## Step 3. Study and check the data

# ### 1. How many events are in the logs?

# In[16]:


data_events=data_foodLogs.event_name.value_counts()

display(data_events)
print('There are {0} events in the logs.'.format(data_events.shape[0]))


# In[17]:


#Code added v.1 
print('There are {0} events in the logs.'.format(data_foodLogs.shape[0]))


# ### 2. How many users are in the logs?

# In[18]:


data_users=data_foodLogs.user_id.nunique()

#display(data_events)
print('There are {0} unique users in the logs.'.format(data_users))


# ### 3. What's the average number of events per user?

# In[19]:


data_user_event=data_foodLogs.groupby('user_id')['event_name'].agg('count').reset_index().sort_values(by=['user_id','event_name'])#agg({'event_name':'count'})
print('The average number of events per user is: ',round(data_user_event.event_name.mean()))

data_user_eventwise=data_foodLogs.groupby(['event_name']).agg({'user_id':['nunique','count']}).reset_index()
data_user_eventwise.columns=['event_name','uniqueUsersCount','totalVisit']
data_user_eventwise['avgVisitPerUsers']=data_user_eventwise.totalVisit/data_user_eventwise.uniqueUsersCount
data_user_eventwise


# #### Conclusion:
# 1. The average no. of events per user are 32.
# 2. If we check event wise average no. of users then found on an average user visits main screen 16 times and then Offer screen, Cart screen and Payment screen. The diffeerence between Offer, Cart and Payment screen are very low which means if users come to offer screen then there is a great posibility he/she visit to cart and then payment screen.
# 3. Tutorial is the least visited screen.

# ### 4. What period of time does the data cover? Find the maximum and the minimum date. Plot a histogram by date and time. Can you be sure that you have equally complete data for the entire period? Older events could end up in some users' logs for technical reasons, and this could skew the overall picture. Find the moment at which the data starts to be complete and ignore the earlier section. What period does the data actually represent?

# In[20]:


min_eventDate=data_foodLogs.event_datetime.min()
max_eventDate=data_foodLogs.event_datetime.max()
print(min_eventDate)
print(max_eventDate)


# In[21]:


sns.set(style="darkgrid")
fig_dims = (15, 5)
fig, axs = plt.subplots(figsize=fig_dims)
sns.histplot(data=data_foodLogs, x="event_datetime", color="skyblue",  kde=True,ax=axs , label='Event Date')
axs.set(xlabel='Event Date', ylabel='No. of records',title='Histogram of Event Date and Time') 
axs.legend() 
for item in axs.get_xticklabels():
    item.set_rotation(45)
fig.show()


# #### Conclusions:
# 1. All the data are from the period between 2019-07-25 and 2019-08-07.
# 2. As we see from the above graph, before 1-August-2019 there is no enough data for the analysis. So we take data from 1-August till 07-Aug.

# There are very small number of events before 1-August. This might be due to various reasons:
# 1. There may be some technical issues on the app prior and it gets resolved in august which incresed the events.
# 1. Company expands/starts its services in other profitable region from august.
# 2. Comapny started its online booking/services from august.
# 3. It has introduced some new menu from august.
# 4. It has invested in advertising before which has started to pay off from august etc.

# ### 5. Did you lose many events and users when excluding the older data?

# In[22]:


data_NotIncluded=data_foodLogs[data_foodLogs['event_datetime']<'2019-08-01']
data_foodLogs_final=data_foodLogs[data_foodLogs['event_datetime']>='2019-08-01']
print('% of data which will not be inlcuded if delete older data are:' ,round(data_NotIncluded.shape[0]/data_foodLogs.shape[0]*100,2))
#Getting users list which we will not inlcude, if delete older data.
data_NotIncludedUsers=data_NotIncluded[~data_NotIncluded.user_id.isin(data_foodLogs_final.user_id)]
display(data_NotIncludedUsers)
print('Total no. of unique users which will not be included if delete older data are:{0} which is almost {1:.2f}% of total unique users'
      .format(data_NotIncludedUsers.user_id.nunique(), 
              (data_NotIncludedUsers.user_id.nunique()/data_foodLogs.user_id.nunique()*100)
             ))


# #### Conclusion:
# 1. % of data which will not be inlcuded if delete older data are: 1.16%
# 2. Total no. of unique users which will not be included if delete older data are 17, which is almost 0.23% of total unique users
# 3. So now we can say from the above calculations that older data are very less in % of all data, so we can delete that and can continue with further analysis.

# ### 6. Make sure you have users from all three experimental groups.

# In[23]:


data_foodLogs_final.groupby('exp_id')['user_id'].nunique()


# #### Conclusion:
# All the experiments contains enough data for analysis.

# ### Conclusion:
# So far, we have loaded the data. Rename, added the new columns as per the convenient. Deleted the duplcate rows. Cheked all the data for any inappropriate data. We found some older data/events which are very less as compare to august data if we include this it can skew our data as well as can impact our analysis. So we decided to remove those outlier first and then will perform the further analysis.

# <a id='step4'></a>
# ## Step 4. Study the event funnel

# ### 1. See what events are in the logs and their frequency of occurrence. Sort them by frequency.

# In[24]:


data_eventFreq=data_foodLogs_final.event_name.value_counts().rename_axis('event_name').reset_index(name='count').sort_values(by='count', ascending=False)
data_eventFreq['%ToTotalData']=round((data_eventFreq['count']/len(data_foodLogs_final)*100),2)
display(data_eventFreq)


# #### Conclusion:
# As we can see from the above table, 'MainScreenAppear' has highest frequency approx. half of total entries. Offerscreen, Cartscreen and Paymentscreen are approx. 19%, 17% and 14%. There is a less '%' of decrease in these three screen as compare to first screen. Tutorial has the least no. of visits i.e. only 0.43%.

# ### 2. Find the number of users who performed each of these actions. Sort the events by the number of users. Calculate the proportion of users who performed the action at least once.

# In[25]:


data_foodEventwiseUser=data_foodLogs_final.groupby('event_name')['user_id'].nunique().rename_axis('event_name').reset_index(name='user_count').sort_values(by='user_count', ascending=False)
#print((data_foodLogs_final.user_id.nunique()))
data_foodEventwiseUser['in_pct']=round((data_foodEventwiseUser.user_count/(data_foodLogs_final.user_id.nunique())*100),2)
display(data_foodEventwiseUser)


# #### Conclusion:
# So here we can say that most of users i.e. 98% of users vists the MainScreenAppear. After that 60% of them visit OffersScreenAppear, 50% CartScreenAppear and 46% visits PaymentScreenSuccessful. only 11% of them visits Tutorial.

# ### 3. In what order do you think the actions took place. Are all of them part of a single sequence? You don't need to take them into account when calculating the funnel.

# After looking at the above table, we can assume the sequence of the action like:
# MainScreenAppear > OffersScreenAppear > CartScreenAppear > PaymentScreenSuccessful.
# Tutorial is optional as users does not require to visit the Tutorial. So we may not inlcude it as part of sequence.
# 
# There are also chances that without visiting Offers screen, user can directly go to the CartScreen or PaymentScreen but in all cases users must visit to the Main Screen and for making purchase Payment screen. 

# ### 4. Use the event funnel to find the share of users that proceed from each stage to the next. (For instance, for the sequence of events A → B → C, calculate the ratio of users at stage B to the number of users at stage A and the ratio of users at stage C to the number at stage B.)

# In[26]:


data_funnel_shift=data_foodEventwiseUser
data_funnel_shift['pct_change']=data_funnel_shift['user_count'].pct_change()
data_funnel_shift


# In[27]:


data_funnel_group=[]
data_foodLogs_final=data_foodLogs_final[data_foodLogs_final.event_name!='Tutorial']
for i in data_foodLogs_final.exp_id.unique():
    df=data_foodLogs_final[data_foodLogs_final.exp_id==i].groupby(['event_name','exp_id'])['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)
   # display(df)
    data_funnel_group.append(df)


# In[28]:


data_funnel_groups=pd.concat(data_funnel_group)
data_funnel_groups


# In[29]:


from plotly import graph_objects as go

fig = px.funnel(data_funnel_groups, x='user_id', y='event_name', color='exp_id',
                title='Displaying users share in each event through Funnel chart ')
fig.show() 


# In[30]:


data_funnel_shift


# #### Conclusion:
# As we can see from the data_funnel_shift table, there is a high % of decrease i.e. 38% in OffersScreen from MainScreen. From Offerscreen to CartScreen it is 18% and from CartScreen to Payment it is only 5%. 
# In Tutorial also 76% decrease but it is not of much interest as this is not required screen to the users.

# ### 5. At what stage do you lose the most users?

# We can say here that we lose most of users at OffersScreen i.e. approx. 38% after Tutorial screen but as Tutorial is not that much inportant to us so we need to concentrate on OffersScreenAppear.

# ### 6. What share of users make the entire journey from their first event to payment?

# In[31]:


df_pvt_minTime=data_foodLogs_final[data_foodLogs_final.event_name != 'Tutorial'].pivot_table(index='user_id', columns='event_name', values='event_datetime', aggfunc='min')
df_pvt_minTime


# In[32]:


df_pvt_minTime=df_pvt_minTime[['MainScreenAppear','PaymentScreenSuccessful']]
df_pvt_minTime=df_pvt_minTime-df_pvt_minTime.shift(+1,axis=1)
df_pvt_minTime


# In[33]:


len(df_pvt_minTime[df_pvt_minTime.PaymentScreenSuccessful.notnull()])/len(df_pvt_minTime)*100


# #### Conclusion:
# Approx. 46% of users make the entire journey from MainScreen to payment screen.

# We can increase the conversion of customer by limiting the loss of customers at each event like at Mainscreen we can display some good/promising offer to the users so that they tends towards next screen.

# <a id='step5'></a>
# ## Step 5. Study the results of the experiment

# ### 1. How many users are there in each group?

# In[34]:


data_ExpWiseUsers=data_foodLogs_final.groupby('exp_id')['user_id'].nunique().reset_index(name='user_count')
data_ExpWiseUsers['in_pct']=round((data_ExpWiseUsers.user_count/sum(data_ExpWiseUsers.user_count)*100),2)
display(data_ExpWiseUsers)


# #### Conclusion:
# Every group has almost same amount of users but group 246 has smallest amount i.e 2484 as compare to other groups.

# ### 2. We have two control groups in the A/A test, where we check our mechanisms and calculations. See if there is a statistically significant difference between samples 246 and 247

# In[35]:


pivot = data_foodLogs_final.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()
pivot


# In[36]:


#Creating function to find statistical significance between groups for particular event.
def check_hypothesis(group1,group2, event, alpha=0.05):
    #let's start with successes, using 
    
    if (group1=='246,247'):# This condition works when we want to compare combined (246,247) group with 248.
        pivot['246_247_combined']=(pivot[246]+pivot[247])
        successes1=pivot[pivot.event_name==event]['246_247_combined'].iloc[0]
        
    else:
        successes1=pivot[pivot.event_name==event][group1].iloc[0]
    successes2=pivot[pivot.event_name==event][group2].iloc[0]
    
    #for trials we can go back to original df or used a pre-aggregated data
    if (group1=='246,247'):# This condition works when we want to compare combined (246,247) group with 248.
        trials1=data_foodLogs_final[data_foodLogs_final.exp_id.isin([246,247])]['user_id'].nunique()
    else:
        trials1=data_foodLogs_final[data_foodLogs_final.exp_id==group1]['user_id'].nunique()
    trials2=data_foodLogs_final[data_foodLogs_final.exp_id==group2]['user_id'].nunique()
    
    #proportion for success in the first group
    p1 = successes1/trials1

   #proportion for success in the second group
    p2 = successes2/trials2

    # proportion in a combined dataset
    p_combined = (successes1 + successes2) / (trials1 + trials2)

  
    difference = p1 - p2
    
    
    z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1/trials1 + 1/trials2))

  
    distr = stats.norm(0, 1) 


    p_value = (1 - distr.cdf(abs(z_value))) * 2

    print('p-value: ', p_value)

    if (p_value < alpha):
        print("Reject H0 for",event, 'and groups',group1,' and ' ,group2)
    else:
        print("Fail to Reject H0 for", event,'and groups',group1,' and ',group2)


# In[37]:


#check_hypothesis(246,247, 'CartScreenAppear', alpha=0.05)


# In[38]:


for evt in data_foodLogs_final.event_name.unique():
    check_hypothesis(246,247, evt, alpha=0.05)


# #### Conclusion:
# **Null Hypothesis H0:** There is no statistically significant difference between samples 246 and 247.<br/>
# **Alternative Hypothesis H1:** There is a statistically significant difference between samples 246 and 247<br/>
# The p-value is greater than significance value which implies 'Fail to Reject H0'. i.e. There is no statistically significant difference between samples 246 and 247 for all events.

# ### 3. Select the most popular event. In each of the control groups, find the number of users who performed this action. Find their share. Check whether the difference between the groups is statistically significant. Repeat the procedure for all other events (it will save time if you create a special function for this test). Can you confirm that the groups were split properly?

# #### Select the most popular event. In each of the control groups, find the number of users who performed this action.

# In[39]:


pivot


# As we can see from the above pivot table, most popular event is MainScreenAppear. It also display number of users for each event and each group. 

# #### Check whether the difference between the groups is statistically significant. Repeat the procedure for all other events.

# In[40]:


check_hypothesis(246,247, 'MainScreenAppear', alpha=0.05)


# In[41]:


for evt in data_foodLogs_final.event_name.unique():
    check_hypothesis(246,247, evt, alpha=0.05)


# #### Conclusion:
# **Null Hypothesis H0:** There is no statistically significant difference between groups.<br/>
# **Alternative Hypothesis H1:** There is a statistically significant difference between groups<br/>
# The p-value is greater than significance value (0.05) which implies 'Fail to Reject H0'. i.e. There is no statistically significant difference between both groups (246 and 247) for all events.

# ### 4. Do the same thing for the group with altered fonts. Compare the results with those of each of the control groups for each event in isolation. Compare the results with the combined results for the control groups. What conclusions can you draw from the experiment?

# In[42]:


for evt in data_foodLogs_final.event_name.unique():
    check_hypothesis(246,248, evt, alpha=0.05)
    check_hypothesis(247,248, evt, alpha=0.05)


# In[43]:


#pivot['246_247_combined']=(pivot[246]+pivot[247])
#pivot


# In[44]:


for evt in data_foodLogs_final.event_name.unique():
    check_hypothesis('246,247',248, evt, alpha=0.05)


# #### Conclusion:
# **Null Hypothesis H0:** There is no statistically significant difference between groups<br/>
# **Alternative Hypothesis H1:** There is a statistically significant difference between groups<br/>
# The p-value is greater than significance value (0.05) which implies 'Fail to Reject H0'. i.e. There is no statistically significant difference between groups. We checked various combinations between the control group and test group like
# 1. 246 and 248
# 2. 247 and 248
# 3. 246 + 247 and 248.
# for all these combination with significance value 0.05, we found no statistically significant difference. thats means all groups are alomost same for all event.

# ### 5. What significance level have you set to test the statistical hypotheses mentioned above? Calculate how many statistical hypothesis tests you carried out. With a statistical significance level of 0.1, one in 10 results could be false. What should the significance level be? If you want to change it, run through the previous steps again and check your conclusions

# #### What significance level have you set to test the statistical hypotheses mentioned above?

# In[45]:


print('We set significance level as 0.05 to test the statistical hypotheses.')


# #### Calculate how many statistical hypothesis tests you carried out.

# There are 5 events and 3 groups so as per this 15 statistical hypothesis tests needs to be carried out.

# #### With a statistical significance level of 0.1, one in 10 results could be false

# In[46]:


for i in pivot.event_name.unique():
    check_hypothesis(246,248, i, alpha=0.01)
    check_hypothesis(247,248, i, alpha=0.01)
    check_hypothesis('246,247',248, i, alpha=0.01)


# #### What should the significance level be?

# In[47]:


#Setting the significance level using Bonferroni correction 
alpha=0.05
noOfHypothesis = 15
bonferroni_alpha = alpha / noOfHypothesis  # three comparisons made
bonferroni_alpha=round(bonferroni_alpha,3)
print(bonferroni_alpha)


# In[48]:


for i in pivot.event_name.unique():
    check_hypothesis(246,248, i, alpha=bonferroni_alpha)
    check_hypothesis(247,248, i, alpha=bonferroni_alpha)
    check_hypothesis('246,247',248, i, alpha=bonferroni_alpha)


# #### Conclusion:
# **Null Hypothesis H0:** There is no statistically significant difference between groups<br/>
# **Alternative Hypothesis H1:** There is a statistically significant difference between groups<br/>
# The p-value is greater than significance value (bonferroni_alpha) which implies 'Fail to Reject H0'. i.e. There is no statistically significant difference between groups. <br/>
# Here we use Bonferroni correction method for setting the alpha value as there are multiple test(15) which we need to run.<br/>
# But if we look at the pValue, This is already greater than existing alpha value(0.05) and now after applying Bonferroni correction it again reduced so find no use to apply this method.<br/>
# As per theory also by lower down the significance level we make more probability of type 2 error and In our case the above test already shows that there is no statistical significant difference between the groups and after decreasing the alpha value it will not change rather it will increase more probability of type 2 error. 

# ## Final Conclusion:
# Below are main points from our observation:
# 1. After loading the data, in the Data Preprocessing task, we found around 413 duplicated value which we dropped otherwise it will impact the analysis and also added two different columns for date and time.
# 2. After plotting histogram, We found some anomalies in the data. There is no enough data before 1-August and this is very less in % of total data approx 1.16% So for analysis we take data from 1-August till 07-Aug only.
# 3. After analyzing the event funnel, we can assume the sequence of the action are like: MainScreenAppear > OffersScreenAppear > CartScreenAppear > PaymentScreenSuccessful. Tutorial is optional as users does not require to visit the Tutorial. So we may not inlcude it as part of sequence. We can also ignore OffersScreenAppear as part of sequence as user can visit directly to Cartscreen or Paymentscreen without visiting the offerscreen.
# 4. There is a high % of decrease in user conversion i.e. 38% in OffersScreen from MainScreen. From Offerscreen to CartScreen it is 18% and from CartScreen to Payment it is only 5%. So we can say there is greater chance of conversion if user also visit to offerscreen from mainscreen so we can recommend our teams to add some most promising offers on the mainscreen to get more conversion from mainscreen.
# 5. Approx. 46% of users are reaching to payment screen which infact is a good number but still we can increase the conversion if can retain customer from Mainscreen to nextscreen.
# 6. In A/A/B testing, we checked various combination of control groups and test groups but found no statistically significant difference between groups. So we can say there is no statistically significant difference between the groups.<br/>
# At last, we can conclude that as there is no difference between the groups conversion after altering the fonts so we can reject the suggeston given by designer.

# In[ ]:




