'''
# Problem Statement

# The average retention rate in the insurance industry is 84%, with the 
# top-performing agencies in the 93%-95% range. 
# Retaining customers is all about the long-term relationship you build.
# Offer a discount on the client's current policy will ensure he/she buys a 
# new product or renews the current policy.
# Studying clients' purchasing behavior to figure out which types of products 
# they're most likely to buy is very essential. 


# CRISP-ML(Q) process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance

'''


'''
1st STEP:
1. Business and Data Understanding :
'''
# Objective(s): Maximize the Sales 
# Constraints: Minimize the Customer Retention


'''Success Criteria'''

# Business Success Criteria: Increase the Sales by 10% to 12% by targeting
#                            cross-selling opportunities for current customers.

# ML Success Criteria: Achieve a Silhouette coefficient of at least 0.6

# Economic Success Criteria: The insurance company will see an increase in 
#                            revenues by at least 8%



'''
data collection
'''
# The auto insurance data is availbale in our lMS website.
#  - dataset contain 9134 customer details 
#  - and 24 feature is recorded for each customer
                   
# data description : 
#    - we have 24 different different features for every customer .
#   - features like customer policies, customer responses, customer name,marital status and education of customers.
        
  
'''
2nd STEP: 
Data preparation (data cleaning)    
'''
import pandas as pd
df = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\DATASETS\AutoInsurance (2).csv") 

# Credentials to connect to sql Database
from sqlalchemy import create_engine
user = 'root'  # user name
pw = 'root'  # password
db = 'univkm_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('univ_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from univ_tbl;'
org_df = pd.read_sql_query(sql, engine)


org_df.shape
org_df.dtypes
org_df.info()
org_df.describe()    # here we see not any null values are present in our dataset.


# checking duplicated rows
org_df.duplicated().sum() # here we get not any duplicated values present in dataframe

#------------------------------------------------------------------------------------------

# now i want to drop some columns which is not good for our clusters (they not give any information for clustering)
# this is nominal dataset.

# Drop the 'customer' column and store the result in 'df_new'
df = org_df.drop(columns=['Customer'])

#df.drop(['Customer'], axis = 1, inplace = True) 

# outlier treatment:
# now we want to see if any outliers are present or not.
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 


from feature_engine.outliers import Winsorizer 
import seaborn as sns

winsor = Winsorizer(capping_method='iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Customer Lifetime Value'])
df['Customer Lifetime Value'] = winsor.fit_transform(df[['Customer Lifetime Value']])
sns.boxplot(df['Customer Lifetime Value'])


winsor = Winsorizer(capping_method ='iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Monthly Premium Auto'])
df['Monthly Premium Auto'] = winsor.fit_transform(df[['Monthly Premium Auto']])
sns.boxplot(df['Monthly Premium Auto'])


winsor = Winsorizer(capping_method='gaussian',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Number of Open Complaints'])
df['Number of Open Complaints'] = winsor.fit_transform(df[['Number of Open Complaints']])
sns.boxplot(df['Number of Open Complaints'])


winsor = Winsorizer(capping_method='gaussian',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Total Claim Amount'])
df['Total Claim Amount'] = winsor.fit_transform(df[['Total Claim Amount']])
sns.boxplot(df['Total Claim Amount']) 

# now we want to check again any outliers are present or not 
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 

# here we succesfully replace outliers as a inliers. 

# now i want to convert categorical data into numerical data :(here we use one hot encoding)
df1 = pd.get_dummies(df, drop_first=True, dtype=int) 


# NORMALIZATION:
# now we normalize our data:
from sklearn.preprocessing import MinMaxScaler # to range between 0 - 1.
from sklearn.pipeline import make_pipeline

cols = list(df1.columns)  
print(cols)

pipe1 = make_pipeline(MinMaxScaler())
normalize = pd.DataFrame(pipe1.fit_transform(df1), columns = cols, index = df1.index)

# here our dataset is normalize





#==============================================================================================================







'''
3rd step:
Model Building (data mining)    
'''
# DBSCAN clustering :density based spatial clustering application with noise
# 1st step =  in this technique we dont want to decide cluster no, in advanced. we only decide eps value and min_samples values.

# checking eps and min_samples value on the basis of silhouette score with our dataset.
# in this operation we get which is best eps and min_samples value for our dataset to make a clusters.


from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score 


z = [ [25, 5], [30, 5], [35, 5],[45, 8], [25, 7], [35, 8]] 
for ep, min_sample in z:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(df1)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(df1, db_clusters))
    
    
# Generate clusters using DBSCAN
model = DBSCAN(eps = 45, min_samples = 8) 
db_clusters = model.fit_predict(df1)


# now check silhouette score 
from sklearn import metrics
metrics.silhouette_score(df1,db_clusters)  


# now save this model:
    # saving dbscan
import pickle
pickle.dump(model, open('dbscan_model.pkl', 'wb'))

model = pickle.load(open('dbscan_model.pkl', 'rb'))


# Concatenate the Results with data
org_df['cluster'] = db_clusters 


# result save in csv file
org_df.to_csv("Insurance_ans.csv",encoding = 'utf-8') 










