#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import time
from warnings import simplefilter,filterwarnings
#simplefilter(action='always',category='FutureWarning')


# In[6]:


train=pd.read_csv('train.csv')
#train=train[train['no_of_trainings']<10]
test=pd.read_csv('test.csv')
train.rename(columns={'KPIs_met >80%':'kpi_met_80'},inplace=True)
test.rename(columns={'KPIs_met >80%':'kpi_met_80'},inplace=True)
train.rename(columns={'awards_won?':'awards_won_prev_year'},inplace=True)
test.rename(columns={'awards_won?':'awards_won_prev_year'},inplace=True)
train.shape,test.shape


# In[7]:


train.head()


# ## Removing NAN

# In[8]:


#filling null values based on correlation
colormap = plt.cm.RdBu
plt.figure(figsize=(9,9))
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
def impute(df):
    df.loc[np.logical_and(df['kpi_met_80']==0 , 
                         df['previous_year_rating'].isna()),'previous_year_rating']=3
    df.loc[np.logical_and(df['kpi_met_80']==1 , 
                         df['previous_year_rating'].isna()),'previous_year_rating']=5
    df['education'].fillna("Bachelor's",inplace=True)
    
impute(train)
impute(test)


# In[9]:


train.describe().round(1)


# In[10]:


def plot_pivots(df):
    for i in ['department','education','gender','recruitment_channel','no_of_trainings',
              'previous_year_rating','kpi_met_80','awards_won_prev_year','age','length_of_service']:
        df.pivot_table(index=i,values='is_promoted').plot.bar()#.sort_values('is_promoted').plot.bar()
        plt.show()
print('train:')
plot_pivots(train)


# ## Removing Correlation

# In[11]:


def corr(df):
    df['expertise']=df['length_of_service']/df['age']*df['length_of_service']
    df.drop(['length_of_service'],axis=1,inplace=True)
corr(train)
corr(test)


# ## Adding features

# In[12]:


def add_features(df):
    df.loc[np.logical_and(df['kpi_met_80']==1,df['awards_won_prev_year']==1),'def_great']=1
    df['def_great'].fillna(0,inplace=True)
    for i in range(2,6):
        df.loc[np.logical_and(df.awards_won_prev_year==1,df.previous_year_rating==i),
           'rated_awards_'+str(i)]=1
        df['rated_awards_'+str(i)].fillna(0,inplace=True)
    
    for i in range(1,5):
        df.loc[np.logical_and(df['kpi_met_80']==1,df['previous_year_rating']==i)
              ,'kp_rating_level'+str(i)]=1
        df['kp_rating_level'+str(i)].fillna(0,inplace=True)
    #df.drop('previous_year_rating',axis=1,inplace=True)
    
add_features(train)
add_features(test)

####Binning
def age_cut(df,bins,labels):
    df['age_cat']=pd.cut(df['age'],bins=bins,labels=labels)
    return df
bins=[19,30,40,50,70]
labels=['young','adults','expert_adults','old']
train=age_cut(train,bins,labels)
test=age_cut(test,bins,labels)


# In[13]:


def create_dummies(df,column_name):
    dummy_table=pd.get_dummies(df[column_name],prefix='Class')
    df=pd.concat([df,dummy_table],axis=1)
    return df
for i in ['department','education','gender','no_of_trainings',
         'kpi_met_80','region','age_cat']:#'region' ,'recruitment_channel'
    train=create_dummies(train,i)
    test=create_dummies(test,i)
train.drop(['Class_Technology',"Class_Master's & above",'Class_m',
           'Class_1','Class_10','Class_region_7','Class_old'],axis=1,inplace=True)
test.drop(['Class_Technology',"Class_Master's & above",'Class_m',
           'Class_1','Class_region_7','Class_old'],axis=1,inplace=True)


# In[14]:


from sklearn.preprocessing import minmax_scale
train['avg_training_score']=minmax_scale(train.avg_training_score)
train['expertise']=minmax_scale(train.expertise)
test['avg_training_score']=minmax_scale(test.avg_training_score)
test['expertise']=minmax_scale(test.expertise)
train['previous_year_rating']=minmax_scale(train.previous_year_rating)
test['previous_year_rating']=minmax_scale(test.previous_year_rating)


# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(15,15))
sns.heatmap(train.iloc[:,6:].corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# ## Modelling

# ## Random Classifier

# In[17]:


skf=StratifiedKFold(n_splits=11,shuffle=True,random_state=42)
predictions1=[]


# In[18]:


from imblearn.over_sampling import SMOTE
columns=train.columns[8:]
columns=columns.drop(['is_promoted','age_cat'])
train_scores,val_scores=[],[]
for train_idx,val_idx in skf.split(train[columns],train.is_promoted):
    train_X,val_X = train.iloc[train_idx,:] , train.iloc[val_idx,:]
    train_y,val_y = train.is_promoted[train_idx],train.is_promoted[val_idx]
    
    tic=time.time()
    sm=SMOTE(random_state=42)
    train_X_rs,train_y_rs=sm.fit_sample(train_X[columns],train_y)
    model=RandomForestClassifier(n_estimators=100, 
                                 min_samples_split=5, 
                                 min_samples_leaf=1, 
                                 max_features=0.7,
                                 max_depth=100, 
                                 bootstrap=True)
    model.fit(train_X_rs,train_y_rs)
    predictions1.append(model.predict(test[columns]))
    train_scores.append(f1_score(model.predict(train_X_rs),train_y_rs))
    print('train : {}'.format(f1_score(model.predict(train_X_rs),train_y_rs)))
    val_scores.append(f1_score(model.predict(val_X[columns]),val_y))
    print('val : {}'.format(f1_score(model.predict(val_X[columns]),val_y)))
print('train score: {}'.format(np.array(train_scores).mean().round(3))
      ,'val score: {}'.format(np.array(val_scores).mean().round(3)))


# ## Catboost

# In[19]:


from imblearn.over_sampling import SMOTE
columns=train.columns[8:]
columns=columns.drop(['is_promoted','age_cat'])
train_scores,val_scores = [],[]
skf=StratifiedKFold(n_splits=11,shuffle=True,random_state=424)
for train_idx,val_idx in skf.split(train[columns],train.is_promoted):
    train_X,val_X = train.iloc[train_idx,:] , train.iloc[val_idx,:]
    train_y,val_y = train.is_promoted[train_idx],train.is_promoted[val_idx]
    
    tic = time.time()
    sm = SMOTE(random_state = 424)
    train_X_rs,train_y_rs = sm.fit_sample(train_X[columns],train_y)
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations = 700,
        learning_rate = 0.08,
        #random_strength=0.1,
        depth = 8,
        loss_function = 'Logloss',
        eval_metric = 'F1',
        metric_period = 100,    
        leaf_estimation_method = 'Newton')
    cb.fit(train_X_rs, train_y_rs,
                 eval_set = (val_X[columns],val_y),
                 #cat_features=categorical_var,
                 use_best_model = True,
                 verbose = True)
    predictions1.append(cb.predict(test[columns]))
    train_scores.append(f1_score(cb.predict(train_X_rs),train_y_rs))
    print('train : {}'.format(f1_score(cb.predict(train_X_rs),train_y_rs)))
    val_scores.append(f1_score(cb.predict(val_X[columns]),val_y))
    print('val : {}'.format(f1_score(cb.predict(val_X[columns]),val_y)))
print('train score: {}'.format(np.array(train_scores).mean().round(3))
      ,'val score: {}'.format(np.array(val_scores).mean().round(3)))


# ## XGBoost

# In[21]:


skf=StratifiedKFold(n_splits=11,shuffle=True,random_state=42)
from xgboost import XGBClassifier
#rain['age_cat_scaled']=train['age_cat_scaled'].astype('float64')
#est['age_cat_scaled']=test['age_cat_scaled'].astype('float64')
columns=train.columns[8:]
columns=columns.drop(['is_promoted','age_cat'])
train_scores,val_scores=[],[]
for train_idx,val_idx in skf.split(train[columns],train.is_promoted):
    train_X,val_X = train.iloc[train_idx,:] , train.iloc[val_idx,:]
    train_y,val_y = train.is_promoted[train_idx],train.is_promoted[val_idx]
    
    tic=time.time()
    sm=SMOTE(random_state=42)
    train_X_rs,train_y_rs=sm.fit_sample(train_X[columns],train_y)
    xg=XGBClassifier(
                learning_rate =0.06,
                 n_estimators=500,
                 max_depth=5,
                 min_child_weight=1,
                 gamma=0,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective= 'binary:logistic',
                 nthread=4,
                 scale_pos_weight=1,
                 seed=27,verbose=True)
    xg.fit(train_X[columns],train_y)
    predictions1.append(xg.predict(test[columns]))
    train_scores.append(f1_score(xg.predict(train_X[columns]),train_y))
    print('train : {}'.format(f1_score(xg.predict(train_X[columns]),train_y)))
    val_scores.append(f1_score(xg.predict(val_X[columns]),val_y))
    print('val : {}'.format(f1_score(xg.predict(val_X[columns]),val_y)))
print('train score: {}'.format(np.array(train_scores).mean().round(3))
      ,'val score: {}'.format(np.array(val_scores).mean().round(3)))


# In[71]:


# Averaging
#weights=[0.1]*11 + [0.45]*11 + [0.45]*11
avg_prediction = []
for j in range(len(predictions1[0])):
    avg_prediction.append(np.mean([prediction[j] for prediction in predictions1]))
#avg_prediction.append(np.mean(np.multiply([prediction[j] for prediction in predictions],weights)))


# In[72]:


for i in range(len(avg_prediction)):
    if avg_prediction[i]>=0.29:
        avg_prediction[i]=1
    else:
        avg_prediction[i]=0


# ## Submission

# In[73]:


holdout_predictions=np.int64(avg_prediction)
#holdout_predictions=[0]*test.shape[0]


# In[74]:


test['holdout']=holdout_predictions
test.holdout.value_counts()


# In[75]:


submission_dict={'employee_id':test['employee_id'],
              'is_promoted':holdout_predictions}
submission_df=pd.DataFrame(data=submission_dict)
submission_df.to_csv('Submission.csv',index=False)

