#!/usr/bin/env python
# coding: utf-8

# ## Project Description
# 
# Our project is to prepare a report for a bank’s loan division. We’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Our report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.

# ### Step 1. Open the data file and have a look at the general information. 

# In[2]:


# Read and import the file in to the dataframe dataCreditScoring
import pandas as pd

#from nltk.stem import SnowballStemmer 
from nltk.stem import SnowballStemmer

dataCreditScoring=pd.read_csv('credit_scoring_eng.csv')

#print general information of the dataframe
dataCreditScoring.info()

#print first 10 rows
print (dataCreditScoring.head())

#print (dataCreditScoring.tail())
#print (dataCreditScoring.sample())


# ### Conclusion

# 1. By using info() method, we can check the table information like columns name, their datatype and count of null values etc. 
# 2. By examine the outcome, we found there are some null values in the column 'days_employed' and 'total_income'. 
# 3. To analyze all data in detail, we have to process the table further. 
# 

# ### Step 2. Data preprocessing

# ### Processing missing values

# In[30]:


#print all columns with null values count.
print(dataCreditScoring.isnull().sum())


# ##### 1. Verifying and updating days_employed

# In[31]:


#Check all columns one by one.
      
###################################### 1. Verifying and updating days_employed ######################################
print('####################### Verifying and updating days_employed column #############################################')
      
#Find % of missing value in days_employed 
missingCount=dataCreditScoring['days_employed'].isnull().sum()
totalRows=len(dataCreditScoring)
print('Approx. {:0.0%} rows are missing'.format(missingCount/totalRows))
      
#check all other available values in the column. 
#We saw there are some -ve and +ve values so now we find how many are +ve and -ve.   
print('Total negative Count: ',dataCreditScoring[dataCreditScoring['days_employed']<0]['days_employed'].count()) 
print('Total positive Count: ',dataCreditScoring[(dataCreditScoring['days_employed']>0)]['days_employed'].count())

#From the +ve values we try to find how many are valid values as normal person works max. for 50 to 60 years.       
print('No. of values fall in between the aprropriate age range of normal human being are: ', dataCreditScoring[(dataCreditScoring['days_employed']>0) & (dataCreditScoring['days_employed']<40000)]['days_employed'].count())
# No values fall in the valid range, so all data is incorrect. 

      
#Now we will also check the Mean and Median
print('The Mean and Median of Days_Employed column are: {0} and {1}'.format(dataCreditScoring['days_employed'].mean(), dataCreditScoring['days_employed'].median()))
# There is huge difference between Mean and Median values. So it's confirmed that there are outliers in this column.
# So before deleting the whole column we will discuss with the team who provide the data first.

############################################### END #######################################################################


# ##### 2. Verifying and updating children

# In[32]:


###################################### 2. Verifying and updating children ###############################################################
print('####################### Verifying and updating children column  #############################################')
#print all children unique values and their count.
print(dataCreditScoring['children'].value_counts(ascending=True))
# found 2 rows as invalid (-1, 47) 

#Update the 2 records as per below condition.      
dataCreditScoring.loc[dataCreditScoring['children']==-1, 'children']=1
dataCreditScoring.loc[dataCreditScoring['children']==20, 'children']=2
      
#Verify the coulmn after updation.
print('After Updation:')
print(dataCreditScoring['children'].value_counts(ascending=True))
#################################################  END ###################################################################


# ##### 3. Verifying and updating dob_years

# In[33]:


########################################## 3. Verifying and updating dob_years  ##########################################################
print('####################### Verifying and updating dob_years column ################################################')
#print all dob_years unique values and their count.
print(dataCreditScoring['dob_years'].value_counts(ascending=True))
#Found there are 101 rows where dob_years is 0 so need to update this
      
#Calculate Mean and Meadian of dob_years
print('The Mean and Median of dob_years column are: {0} and {1}'.format(dataCreditScoring['dob_years'].mean(), dataCreditScoring['dob_years'].median()))
#The means and medians are roughly equal, which implies that there are no outliers in the Days_Employed columns.
      
#now we also check mean and median by gender column 
print(dataCreditScoring.groupby('gender')['dob_years'].agg(['mean','median']))
#there is no such big differences in the mean and medain values based on gender.
#so we can use median value to fill the days_employed column where dob_years=0

#Update missing values
dataCreditScoring.loc[dataCreditScoring['dob_years']==0, 'dob_years']=dataCreditScoring['dob_years'].median()

#Verify the coulmn after updation.
print('After Updation:')
print(dataCreditScoring['dob_years'].value_counts(ascending=True))
################################################ End ######################################################################


# ##### 4. Verifying and updating education

# In[34]:


########################################## 4. Verifying and updating education  ##########################################################
print('####################### Verifying and updating education  #######################################################')
#print all education unique values and their count.
print(dataCreditScoring['education'].value_counts(ascending=True)) 
#All values seems fine except 1 data i.e. graduate degree and bachelor's degree. Both are same so we update any one of them with other.

#Also update all values to lower case
dataCreditScoring['education'] = dataCreditScoring['education'].str.lower()

#Replacing graduate degree with bachelor's degree.
dataCreditScoring.loc[dataCreditScoring['education']=='graduate degree', 'education']="bachelor's degree"

#Verify the coulmn after updation.
print('After Updation:')
print(dataCreditScoring['education'].value_counts(ascending=True))
################################################## End  ###############################################################


# ##### 5. Verifying and updating education_id 

# In[35]:


########################################### 5. Verifying and updating education_id  ######################################
print('####################### Verifying and updating education_id column')
#print all education_id unique values and their count.
print(dataCreditScoring['education_id'].value_counts(ascending=True))# all data seems correct
########################################## End ############################################################################


# ##### 6. Verifying and updating family_status

# In[36]:


########################################### 6. Verifying and updating family_status  ##########################################################
print('####################### Verifying and updating family_status column')

#print all family_status unique values and their count.
print(dataCreditScoring['family_status'].value_counts(ascending=True))

# updated all values to lower case
dataCreditScoring['family_status'] = dataCreditScoring['family_status'].str.lower()

#Verify the coulmn after updation.
print('After Updation:')
print(dataCreditScoring['family_status'].value_counts(ascending=True))
############################################# End  ########################################################################


# ##### 7. Verifying and updating family_status_id 

# In[37]:


########################################### 7. Verifying and updating family_status_id  ##########################################################
print('####################### Verifying and updating family_status_id column')
#print all family_status unique values and their count.
print(dataCreditScoring['family_status_id'].value_counts(ascending=True))# all data seems fine
############################################################################################################################


# ##### 8.  Verifying and updating gender

# In[38]:


########################################## 8.  Verifying and updating gender  ##########################################################
print('####################### Verifying and updating gender column')

#print all gender unique values and their count.
print(dataCreditScoring['gender'].value_counts(ascending=True))
#Found 1 record with the value 'XNA' but seems someone don't wants to mention the gender so so we keep this as it is.
print(dataCreditScoring[dataCreditScoring['gender']=='XNA'])

############################################### End #######################################################################


# ##### 9. Verifying and updating income_type 

# In[39]:


######################################### 9. Verifying and updating income_type  ##########################################################
print('####################### Verifying and updating income_type column')

#print all income_type unique values and their count.
print(dataCreditScoring['income_type'].value_counts(ascending=True))
dataCreditScoring['income_type'] = dataCreditScoring['income_type'].str.lower()#just update all values in to lower case

################################################ End ######################################################################


# ##### 10. Verifying and updating debt  

# In[40]:


########################################## 10. Verifying and updating debt  ##########################################################
print('####################### Verifying and updating debt column')
#print all debt unique values and their count.
print(dataCreditScoring['debt'].value_counts(ascending=True))#all seems correct

############################################# End #########################################################################


# ##### 11. Verifying and updating total_income

# In[41]:


######################################### 11. Verifying and updating total_income  ##########################################################
print('####################### Verifying and updating total_income column')
#print all total_income unique values and their count.
print(dataCreditScoring['total_income'].value_counts(ascending=True))
#All seems correct except for the null values. total no of null records are exact same as of days_employed column. which is logically correct.  

####################################################### End ##############################################################


# ##### 12. Verifying and updating purpose 

# In[42]:


###########################################  12. Verifying and updating purpose  ##########################################################
print('####################### Checking and Updating values in purpose column')
#update all values to lower
dataCreditScoring['purpose']=dataCreditScoring['purpose'].str.lower()
#print all total_income unique values and their count.
print(dataCreditScoring['purpose'].value_counts(ascending=True))
# Data is fine but We need to categorize the purpose data which we will do in categorizing section.
########################################End#############################################################################


# ## Conclusion

# 1. First checked all the coulmns for null value. Found null values in days_employed and total_income columns.
# 2. Start analysis all column one by one. First start with days_employed column.
# 3. days_employed: Around 10% data is missing and the other available data is also not valid, So before deleting the whole column we will discuss with the team who provide the data. 
# 
# 4. children : Found 2 incorrect data (-1,20). Replace the incorrect values with correct values(1,2) as children can't be -1 and 20.
# 
# 5. dob_years : There are around 101 rows where values is 0, and rest of the values seems fine. After analysing we found that we can replace 0 with median values. So we updated the missing values by median. 
# 
# 6. education : All values seems fine except 1 data i.e. 'graduate degree' and 'bachelor's degree'. As both are same so we update any one of them with the other.
# 
# 7. education_id : All data seems correct.
# 8. family_status : Just updated all values to lower case.
# 9. gender : Found 1 record with the value 'XNA' but may be someone don't wants to mention the gender so so we keep this as it is.
# 
# 10. income_type: Just updated all values to lower case.
# 11. debt: All seems correct.
# 12. total_income: All seems correct except for the null values. Total no. of null records are exact same where days_employed is also null which is logically correct.  
# 13. purpose: Updated all values to lower. Data is fine but we need to categorize the purpose data which we will do in categorizng section.
# 
#  
# So far we have checked all the data. Except days_employed columns we have processed all columns missing values.
# 
# 

# <div><b> Removing duplicates from total_income:</b>
#    <ol> 
#        <li>First check total null values count of 'days_employed' and 'total_income' column </li>
#        <li>As we checked above also, data in days_employed column is invalid and also we don't need it in our further analysis, so we will leave this column as it is for now.</li>
#        <li>Finally we fill total_income null values by the median value. To calculate median we first group the total_income by income_type and then calculate median.</li>
#     </ol>
# </div>

# In[43]:


#checking for none values in total_income column
print(dataCreditScoring['total_income'].isnull().sum())
#print(dataCreditScoring.groupby('income_type')['total_income'].agg(['mean','median','sum']))

#Replacing the null values.
dataCreditScoring['total_income']=dataCreditScoring.groupby('income_type')['total_income'].transform(
    lambda x: x.fillna(x.median())
    )
#print(dataCreditScoring['total_income'].isnull().sum())


# ### Data type replacement

# In[44]:


#Update Data Types of some columns. 
dataCreditScoring['children'] = dataCreditScoring['children'].astype('int8')
dataCreditScoring['dob_years'] = dataCreditScoring['dob_years'].astype('int8')
dataCreditScoring['education_id'] = dataCreditScoring['education_id'].astype('int16')
dataCreditScoring['family_status_id'] = dataCreditScoring['family_status_id'].astype('int16')
dataCreditScoring['education'] = dataCreditScoring['education'].astype('str')
#verify the updation.
dataCreditScoring.info()


# ### Conclusion

# After analyzing the data we also found there are some columns for which we can replace data type. like for children column we don't need int64 data type. So we change the data types to int8.
# 
# We can also see the impact of doing so by executing info(). Now after replacing data types, memory usage has dropped to 1.4 from 2.0. thats looks great.

# ### Processing duplicates

# In[45]:


#printing the count of all duplicate records
print('No. of duplicate records : {0}'.format(dataCreditScoring.duplicated().sum()))
#found 72 duplicate records. 

#deletes all duplicates 
print('Deleted duplicate records.')
dataCreditScoring=dataCreditScoring.drop_duplicates().reset_index(drop=True) 

#verify after deleting the duplicates records.
print('Duplicate records left : {0}'.format(dataCreditScoring.duplicated().sum()))

dataCreditScoring.info()


# ### Conclusion

# So now we have deleted the duplicate records but if we see the purpose column we can easily see some purpose are similar in nature but may be due to their text it is recorded as different or the same person has applied for another loan for same purpose.
# Anyways our task is to calculate whether the customer is defaulted on a loan based on their marital status and number of children, So in our next task we will categorize purpose column to common type and then delete all duplicate rows.

# ### Categorizing Data

# In[46]:


#create a function which will take purpose as input parameter and return some common purpose text which we will add as
#separate column in the the dataframe.
#from nltk.stem import SnowballStemmer 
english_stemmer = SnowballStemmer('english')
def define_Purpose(dataPurpose):
    for word in dataPurpose.split(' '):
        try:
            
            stemmed_word = english_stemmer.stem(word)
            #print(stemmed_word)
            if ((stemmed_word =='educ') | (stemmed_word =='univers')):
                return 'education'
            if stemmed_word =='car':
                return 'car'
            if stemmed_word =='hous':
                return 'house'
            if stemmed_word =='wed':
                return 'wedding'
            if ((stemmed_word =='estat') | (stemmed_word =='properti')):
                return 'real estate'
        except:
            return word
            

#aaa=define_Purpose('wedding is a special event')
#print(aaa)
try:
    
    #calling 'define_Purpose' function and add separate column 'purpose_main' in the dataframe for new purpose.
    dataCreditScoring['purpose_main']=dataCreditScoring['purpose'].apply(define_Purpose)

    #now check duplicate records using 'purpose_main' column and excluding 'purpose' column.
    print('Duplicate records count : {0}'.format(dataCreditScoring[['children','days_employed','dob_years','education','education_id','family_status','family_status_id','gender','income_type','debt','total_income','purpose_main']].duplicated().sum()))
    #there are 252 duplicate rows, so we will delete all those.
   
    #Add dataCreditScoring_ID column and create new dataframe with dataCreditScoring_ID and purpose column so that we have 
    #back up of old purpose before adding the new purpose.
    dataCreditScoring['dataCreditScoring_ID']=dataCreditScoring.index+1
    dataCreditScoring_purpose=dataCreditScoring[['dataCreditScoring_ID','purpose']]
   
    #After taking back up now drop old purpose column 
    dataCreditScoring.drop(['purpose'], axis=1,inplace=True)

    #deletes all duplicates 
    dataCreditScoring=dataCreditScoring.drop_duplicates(subset=['children','days_employed','dob_years','education','education_id','family_status','family_status_id','gender','income_type','debt','total_income','purpose_main'], keep='first')
    print('Deleted duplicate reords')
except:
    print('Already executed code.')
#Print duplicates records again to verify that no duplicates records left
print('Duplicate records left : {0}'.format(dataCreditScoring[['children','days_employed','dob_years','education','education_id','family_status','family_status_id','gender','income_type','debt','total_income','purpose_main']].duplicated().sum()))

#verify the updation.
dataCreditScoring.info()


# ### Conclusion

# In this step, we have categorize the purpose column. By categorizing we get 252 duplicate rows which we have deleted.
# So far we have updated the missing values, changed the required data types and deleted the duplicated rows. 
# Now we got the data for analysing.

# ### Step 3. Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

# In[47]:


#create function which will return Nodefaulter/defaulter based on isdebt column.
def define_Debt(isdebt):
        try:
            if isdebt==0:
                return 'Nodefaulter'
            if isdebt==1:
                return 'defaulter'
        except:
            return word
        
#added new column 'isDefaulter' to the dataframe.
dataCreditScoring['isDefaulter']=dataCreditScoring['debt'].apply(define_Debt)

#Creating pivot table to present the result in particulare format.
data_pivot = dataCreditScoring.pivot_table(index='children', columns='isDefaulter', values='debt', aggfunc='count')
data_pivot['% of defaulter'] = (data_pivot['defaulter']/(data_pivot['Nodefaulter']+data_pivot['defaulter'])*100).round(2)
print(data_pivot.sort_values('% of defaulter')) 


# ### Conclusion

# Analysis are as follows:
# 1. Maximum people who take loan belongs to the category of those who don't have children. May be generally people take loan in their ealry age also if we see the % of defaulter it is the least among all. So people pays loan on time when they don't have kids.
# 
# 2. If we move to next rows i.e. more children means more probability of defaulter but with 5 children nobody is dafaulter. 
# 

# - Is there a relation between marital status and repaying a loan on time?

# In[48]:


#Creating pivot table to present the result in particulare format.
data_pivot_family = dataCreditScoring.pivot_table(index='family_status', columns='isDefaulter', values='debt', aggfunc='count')
data_pivot_family['% of defaulter'] = (data_pivot_family['defaulter']/(data_pivot_family['Nodefaulter']+data_pivot_family['defaulter'])*100).round(2)
print(data_pivot_family.sort_values('% of defaulter')) 


# ### Conclusion

# 1. Widow / widower category's people generally pays loan on time.
# 2. Even Divorced people's '% of defaulter' is less than Married, civil partnership and unmarried.
# 3. Most of the person belongs to Married category and are still less in '% of defaulter' than civil partnership and unmarried.

# - Is there a relation between income level and repaying a loan on time?

# In[49]:


#We’ll count how many defaulters of each income there are using the value_counts() method:
#pd.set_option('display.max_rows', None)
#print(dataCreditScoring['total_income'].sort_values())
#based on the data we create some group based on income level

#Create function which will return income group based on total_income.
def define_income_group(income):
    try:
        if income <= 30000:
            return '0-30k'
        if ((income >30000) &(income <=80000)):
            return '30k-50k'
        if ((income >80000) &(income <=120000)):
            return '80k-120k'
        if ((income >120000) &(income <=200000)):
            return '120k-200k'
        if ((income >200000) &(income <=300000)):
            return '200k-300k'
        if income > 300000:
            return 'more than 300k'
        return 'None' 
    except:
        return 'found error in income'

#print(define_income_group(125832.259))
    
#added new column 'income_group' to the dataframe
dataCreditScoring['income_group']=dataCreditScoring['total_income'].apply(define_income_group)
#print(dataCreditScoring.head(20))# print to check if new column added successfully

#creating pivot table.
data_pivot_income = dataCreditScoring.pivot_table(index='income_group', columns='isDefaulter', values='debt', aggfunc='count')
data_pivot_income['% of defaulter'] = (data_pivot_income['defaulter']/(data_pivot_income['Nodefaulter']+data_pivot_income['defaulter'])*100).round(2)
print(data_pivot_income.sort_values('% of defaulter')) 


# ### Conclusion

# 1. No such relationship found between the income group and % of defaulter 
# 2. The people with highest level income group are mostly defaulter but it's just 1 record there for that category people.

# - How do different loan purposes affect on-time repayment of the loan?

# In[50]:


data_pivot_purpose = dataCreditScoring.pivot_table(index='purpose_main', columns='isDefaulter', values='debt', aggfunc='count')
data_pivot_purpose['% of defaulter'] = (data_pivot_purpose['defaulter']/(data_pivot_purpose['Nodefaulter']+data_pivot_purpose['defaulter'])*100).round(2)
print(data_pivot_purpose.sort_values('% of defaulter')) 


# ### Conclusion

# 1. People who take loan for house generally pays loan on time.
# 2. Those who take loan for education or car are highest in terms of '% of defaulter'

# ### Step 4. General conclusion

# My General Analysis is as follows:
# 1. People whose marital status are among ('widow/widower, divorced, married) and having no kids will have more probabily to pay loan on time.
# 2. As whole data in 'days_employed' column are invalid so we will get some more detail from the team before deleting the whole column.

# ### Overall conclusion
# <br/>
# <div>After analysing the final data with the help of pivot table, we got following observation:
#     <br/>
# <ol>
#     <li>People with no kids generally pays loan on time.</li>
# <li>If the person who is applying for the loan fall under the category (Widow / widower, Divorced, Married) generally pays loan on time. Defaulters are high for 'civil partnership and unmarried people' category people.</li>
# <li>People who take loan for house purpose generally pays loan on time.</li>
# </ol>  
# </div>

# ### Project Readiness Checklist
# 
# Put 'x' in the completed points. Then press Shift + Enter.

# - [x]  file open;
# - [x]  file examined;
# - [x]  missing values defined;
# - [x]  missing values are filled;
# - [x]  an explanation of which missing value types were detected;
# - [x]  explanation for the possible causes of missing values;
# - [x]  an explanation of how the blanks are filled;
# - [x]  replaced the real data type with an integer;
# - [x]  an explanation of which method is used to change the data type and why;
# - [x]  duplicates deleted;
# - [x]  an explanation of which method is used to find and remove duplicates;
# - [x]  description of the possible reasons for the appearance of duplicates in the data;
# - [x]  data is categorized;
# - [x]  an explanation of the principle of data categorization;
# - [x]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [x]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [x]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [x]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [x]  conclusions are present on each stage;
# - [x]  a general conclusion is made.

# In[ ]:




