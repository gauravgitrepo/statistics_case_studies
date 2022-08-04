#!/usr/bin/env python
# coding: utf-8

# In[42]:


import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, plot_roc_curve, plot_confusion_matrix, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.svm import SVC
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# In[39]:


conda install -c conda-forge xgboost


# In[5]:


raw_df = pd.read_excel('creditcard_dataset.xls',header=1)
raw_df


# In[6]:


raw_df.columns


# In[7]:


#Attribute Information:
#
#This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
#LIMIT_BAL: Amount of the given credit (NT dollar): 
#        it includes both the individual consumer credit and his/her family (supplementary) credit.
#SEX: Gender (1 = male; 2 = female).
#EDUCATION: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#MARRIAGE: Marital status (1 = married; 2 = single; 3 = others).
#AGE: Age (year).
#X6 - X11:
#History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows:
#X6 = the repayment status in September, 2005;
#X7 = the repayment status in August, 2005; . . .;
#X11 = the repayment status in April, 2005. 
#The measurement scale for the repayment status is:
#    -1 = pay duly; 
#    1 = payment delay for one month; 
#    2 = payment delay for two months; . . .; 
#    8 = payment delay for eight months; 
#    9 = payment delay for nine months and above.
#X12-X17: Amount of bill statement (NT dollar). 
#    X12 = amount of bill statement in September, 2005;
#    X13 = amount of bill statement in August, 2005; . . .;
#    X17 = amount of bill statement in April, 2005.
#X18-X23: Amount of previous payment (NT dollar). 
#    X18 = amount paid in September, 2005; 
#    X19 = amount paid in August, 2005; . . .;
#    X23 = amount paid in April, 2005.


# In[8]:


raw_df.dtypes


# In[9]:


#Data Pre-Processing
#Creating a calculated field to find Percentage of bill amount paid
raw_df['PERCENTPAY_1'] = round(raw_df['PAY_AMT1']/raw_df['BILL_AMT1'], 2)
raw_df['PERCENTPAY_2'] = round(raw_df['PAY_AMT2']/raw_df['BILL_AMT2'], 2)
raw_df['PERCENTPAY_3'] = round(raw_df['PAY_AMT3']/raw_df['BILL_AMT3'], 2)
raw_df['PERCENTPAY_4'] = round(raw_df['PAY_AMT4']/raw_df['BILL_AMT4'], 2)
raw_df['PERCENTPAY_5'] = round(raw_df['PAY_AMT5']/raw_df['BILL_AMT5'], 2)
raw_df['PERCENTPAY_6'] = round(raw_df['PAY_AMT6']/raw_df['BILL_AMT6'], 2)

raw_df = raw_df.rename(columns = {'default payment next month': 'default'})

raw_df = raw_df.astype({'SEX': 'category', 'EDUCATION': 'category', 'MARRIAGE': 'category', 'PAY_0': 'category', 'PAY_2': 'category', 'PAY_3': 'category',
                 'PAY_4': 'category', 'PAY_5': 'category', 'PAY_6': 'category', 'default': 'category'})

raw_df.dtypes
raw_df.shape


# In[10]:


#Filling all NA values with 1 as PERCENTPAY_X variable is 100% if BILL_AMTX is 0

raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
values = {'PERCENTPAY_1': 1, 'PERCENTPAY_2': 1, 'PERCENTPAY_3': 1, 'PERCENTPAY_4': 1, 'PERCENTPAY_5': 1, 'PERCENTPAY_6': 1}
raw_df = raw_df.fillna(value=values)
raw_df.dtypes


# In[11]:


round(raw_df.groupby('default').size()/len(raw_df), 3)
sns.countplot(x = 'default', 
              data = raw_df)


# In[12]:


display(raw_df.groupby('default').size())


# In[15]:


#Calculate proportion of credit default by age, sex, education and marital status
plotDat = (raw_df.
           groupby(['AGE', 'SEX', 'EDUCATION', 'MARRIAGE']).
           apply(lambda x: len(x.default[x.default == 1])/len(x.default)).
           reset_index().
           rename(columns = {0: 'proportionDefaulted'})
          )
plotDat.sample(5)


# In[16]:


# Visualizing credit default proportion with respect to age and education level
plt.figure(figsize = (20, 10));

sns.relplot(
    x = 'AGE',
    y = 'proportionDefaulted',
    s = 150,
    hue = 'SEX',
    col = 'EDUCATION',
    kind = 'scatter',
    data = plotDat);


# In[17]:


# Data processing for proportion of credit default with respect to marital status
plotDat = raw_df.groupby(['MARRIAGE', 'default'], as_index=False).size()
plotDat = plotDat.pivot(index = 'MARRIAGE', columns = ['default'], values = 'size')
plotDat = plotDat.rename_axis(None).rename_axis(None, axis=1)
plotDat.columns = pd.Index(list(plotDat.columns))
plotDat = plotDat.reset_index(drop=True)
plotDat[2] = plotDat[1]/(plotDat[0]+plotDat[1])
plotDat = plotDat.reset_index()
plotDat = plotDat.rename(columns = { 'index' : 'Marriage', 0 : 'nonDefault', 1 : 'default', 2: 'proportion'})
plotDat = plotDat[1:4]
plotDat


# In[18]:


# Visualizing proportion of credit defaults by marital status
sns.barplot(
  x = 'Marriage',
  y = 'proportion',
  data = plotDat
)


# In[19]:


# Visualizing Limit Balance  
plotDat = (raw_df.groupby(['LIMIT_BAL']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'}))

plotDat.sample(5)

sns.lineplot(
    x = 'LIMIT_BAL',
    y = 'proportionDefaulted',
    data = plotDat);


# In[20]:


# Visualizing correlation between variables
plt.figure(figsize = (20, 10))
corrMatrix = raw_df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[22]:


# three main effects

mod1 = smf.glm(formula = 'default ~ (PERCENTPAY_1 + PERCENTPAY_2 + PERCENTPAY_3 + PERCENTPAY_4 + PERCENTPAY_5 + PERCENTPAY_6)'   
                 , data = raw_df, family = sm.families.Binomial()).fit();

mod2 = smf.glm(formula = 'default ~ (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6)'   
                 , data = raw_df, family = sm.families.Binomial()).fit();

mod3 = smf.glm(formula = 'default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE'   
                 , data = raw_df, family = sm.families.Binomial()).fit();


# In[23]:


#Identifying the best model
models = pd.DataFrame({
    'models': ['mod1', 'mod2', 'mod3'],
    'df': [mod1.df_model, mod2.df_model, mod3.df_model],
    'AIC': [mod1.aic, mod2.aic, mod3.aic]
})

models['deltaAIC'] = models.AIC - min(models.AIC)

models.sort_values('AIC')


# In[24]:


#Output of the best model
mod3.wald_test_terms()


# In[25]:


round(mod1.wald_test_terms().table, 3)


# In[26]:


mod3.summary()


# In[28]:


#Splitting dataset into test and training for Supervised Machine Learning
Train, Test = train_test_split(raw_df, 
                               stratify = raw_df.default,
                               test_size = 0.20, 
                               random_state = 345)

display(len(Train),
        len(Test))

display(Test.groupby('default').size())
round(Test.groupby('default').size()/len(Test), 3)


# In[29]:


#Downsampling data to balance the dataset
np.random.seed(345) # to create reproducible results

maj_class = np.where(Train.default == 0)[0]
min_class = np.where(Train.default == 1)[0]

resample = np.random.choice(maj_class, size = len(min_class), replace = False)

TrainDS = pd.concat([Train.iloc[min_class], Train.iloc[resample]])

TrainDS.shape


# In[30]:


display(TrainDS.groupby('default').size())
round(TrainDS.groupby('default').size()/len(TrainDS), 3)


# In[31]:


#Encoding categorical variables
enc = OneHotEncoder(handle_unknown = 'ignore', 
                    sparse = False)

enc_raw_data_train = TrainDS.select_dtypes(['object', 'category']).drop(columns = 'default') 

enc = enc.fit(enc_raw_data_train)

encoded_data_train = pd.DataFrame(enc.transform(enc_raw_data_train))

encoded_data_train.columns = enc.get_feature_names(enc_raw_data_train.columns)
encoded_data_train.index = enc_raw_data_train.index

TrainDS_Enc = pd.concat([TrainDS.drop(enc_raw_data_train.columns, axis = 1), encoded_data_train], axis = 1)

TrainDS_Enc.head()


# In[32]:


#Removing columns with no Variance
selector = VarianceThreshold()

sel_raw_data_train = TrainDS_Enc.drop(columns = 'default') 

selector = selector.fit(sel_raw_data_train)

selected_data_train = pd.DataFrame(selector.transform(sel_raw_data_train))

selected_data_train.columns = sel_raw_data_train.columns
selected_data_train.index = sel_raw_data_train.index

TrainDS_EncSel = pd.concat([TrainDS_Enc.drop(sel_raw_data_train.columns, axis = 1), selected_data_train], axis = 1)

TrainDS_EncSel.head()


# In[33]:


#Scaling the dataset

Scaler = StandardScaler()

scl_raw_data_train = TrainDS_EncSel.drop(columns = 'default') 

Scaler = Scaler.fit(scl_raw_data_train)

scaled_data_train = pd.DataFrame(Scaler.transform(scl_raw_data_train))

scaled_data_train.columns = scl_raw_data_train.columns
scaled_data_train.index = scl_raw_data_train.index

TrainDS_EncSelScl = pd.concat([TrainDS_EncSel.drop(scl_raw_data_train.columns, axis = 1), scaled_data_train], axis = 1)

TrainDS_EncSelScl.head()


# In[34]:


#encoding test dataset
enc_raw_data_test = Test.select_dtypes(['object', 'category']).drop(columns = 'default') 

encoded_data_test = pd.DataFrame(enc.transform(enc_raw_data_test))

encoded_data_test.columns = enc.get_feature_names(enc_raw_data_test.columns)
encoded_data_test.index = enc_raw_data_test.index

Test_Enc = pd.concat([Test.drop(enc_raw_data_test.columns, axis = 1), encoded_data_test], axis = 1)

Test_Enc.head()

#removing columns with 0 variance in test dataset
sel_raw_data_test = Test_Enc.drop(columns = 'default') 

selected_data_test = pd.DataFrame(selector.transform(sel_raw_data_test))

selected_data_test.columns = sel_raw_data_test.columns
selected_data_test.index = sel_raw_data_test.index

Test_EncSel = pd.concat([Test_Enc.drop(sel_raw_data_test.columns, axis = 1), selected_data_test], axis = 1)

Test_EncSel.head()

#scaling data in test dataset
scl_raw_data_test = Test_EncSel.drop(columns = 'default') 

scaled_data_test = pd.DataFrame(Scaler.transform(scl_raw_data_test))

scaled_data_test.columns = scl_raw_data_test.columns
scaled_data_test.index = scl_raw_data_test.index

Test_EncSelScl = pd.concat([Test_EncSel.drop(scl_raw_data_test.columns, axis = 1), scaled_data_test], axis = 1)

Test_EncSelScl.head()


# In[35]:


# set up data and labels
X_train = TrainDS_EncSelScl.drop(columns = 'default')
y_train = TrainDS_EncSelScl.default

X_test = Test_EncSelScl.drop(columns = 'default')
y_test = Test_EncSelScl.default

y_test.dtypes


# In[43]:


# set up data and labels
X_train = TrainDS_EncSelScl.drop(columns = 'default')
y_train = TrainDS_EncSelScl.default

X_test = Test_EncSelScl.drop(columns = 'default')
y_test = Test_EncSelScl.default

pos_label = 1

# set up scoring metric
scoring_metric = 'balanced_accuracy'

# set up classifiers and tuning parameters
names = [  'Random Forest','XGBoost','MLPClassifier'] #, 'Linear SVM' 'XGBoost' 'Decision Tree', 'AdaBoost', , 
classifiers = [ #DecisionTreeClassifier(random_state = 345), 
               #AdaBoostClassifier(random_state = 345), 
               RandomForestClassifier(random_state = 345),
               XGBClassifier(random_state = 345),
               MLPClassifier(random_state=345)
               #SVC(random_state = 345, kernel = 'linear', probability = True) #,
              
              ]
param_grids = [#{'max_depth': [2, 3, 4, 5], 'criterion': ['gini', 'entropy']}, 
               #{'n_estimators': [100,200,300, 400]}, 
               {'max_depth': [ 8,9, 10], 'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [250,300, 350]}, 
               {'n_estimators': [ 340, 350, 360]},
               {'alpha':[1e-05], 'hidden_layer_sizes': [10,2], 'solver':['lbfgs'], 'max_iter':[1150]}
               #{'C': [0.01,0.05, 0.1, 0.5, 1]}
              ]

# create empty lists for storing outcomes
models = []
preds = []
probs = []
BAs = []
AUCs = []
FPRs = []
TPRs = []
timings = []

# train classifiers and generate test predictions/probabilities
for i, eachClassifier in enumerate(classifiers):
    
    print('Now working on model ', i + 1, ' of ', len(classifiers), ': ', names[i], sep = '')
    
    # define cross-validation/parameter tuning settings
    search = GridSearchCV(eachClassifier, 
                          param_grids[i], 
                          cv = 5, 
                          scoring = scoring_metric,
                          n_jobs = -1) 
    
    model = search.fit(X_train, y_train)
    pred = search.predict(X_test)
    prob = search.predict_proba(X_test)[:, 1]
    
    models.append(model)
    preds.append(pred)
    probs.append(prob)
    
    BAs.append(balanced_accuracy_score(y_test, pred))
    AUCs.append(roc_auc_score(y_test, prob))
    
    FPR, TPR, _ = roc_curve(y_test, prob, pos_label = pos_label)
    
    FPRs.append(FPR)
    TPRs.append(TPR)
    
    timings.append(model.refit_time_)
    
print('Finished!')


# In[45]:


results = pd.DataFrame({'Classifier': names, 
                        'Balanced Accuracy': BAs, 
                        'AUC': AUCs, 
                        'TPR': TPRs, 
                        'FPR': FPRs, 
                        'Refit Time': timings}).sort_values('AUC', ascending = False)

display(round(results[['Classifier', 'Refit Time', 'Balanced Accuracy', 'AUC']], 3))


# In[46]:


best_index = results.index[0]
models[best_index].best_estimator_


# In[47]:


#Plotting ROC curves
def Custom_ROC_Plot (results, X_test, y_test, title):

    fig, ax = plt.subplots(figsize = (8, 8))

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls = '--', c = 'k')
    ax.set(title = title)

    for i in results.index:
        plot_roc_curve(models[i], 
                       X_test, 
                       y_test, 
                       color = cm.Set1(i), 
                       label = results.loc[i, 'Classifier'] + ': ' + str(round(results.loc[i, 'AUC'], 3)), 
                       ax = ax)
        
    return([fig, ax])

fig, ax = Custom_ROC_Plot(results, X_test, y_test, title = 'Test AUC Comparison')


# In[48]:


#Identyfing most important predictors
best_index = results.index[0]
# for models with feature importances
if hasattr(models[best_index].best_estimator_, 'feature_importances_'):
    var_imp = pd.DataFrame({
        'feature': X_test.columns, 
        'importance': models[best_index].best_estimator_.feature_importances_}).sort_values('importance', ascending = False)
# for models with coefficients (directional)
elif hasattr(models[best_index].best_estimator_, 'coef_'):
    var_imp = pd.DataFrame({
        'feature': X_test.columns, 
        'importance_abs': abs(models[best_index].best_estimator_.coef_[0]), 
        'importance': models[best_index].best_estimator_.coef_[0]}).sort_values('importance_abs', ascending = False)
    
sns.barplot(y = 'feature', 
            x = 'importance', 
            data = var_imp.head(10))


# In[49]:


best_index = results.index[0]
tn, fp, fn, tp = confusion_matrix(y_test, preds[best_index]).ravel()
accuracy = round((tp+tn)/(tp+tn+fp+fn),3)
sensitivity = round(tp/(tp+fn),3)
specificity = round(tn/(tn+fp),3)
baccuracy = round((sensitivity+specificity)/2,3)
print("accuracy: ", accuracy)
print("balanced accuracy: ", baccuracy)
print("sensitivity: ", sensitivity)
print("specificity: ", specificity)


# In[50]:


#plotting confusion matrix
plot_confusion_matrix(models[best_index], 
                      X_test, 
                      y_test,
                      cmap = plt.cm.Blues,
                      values_format = 'd')


# In[54]:


#Visualizing monthly repayment status(whether repayment occurred on time or delayed) versus credit default rate

plotDat0 = (raw_df.groupby(['PAY_0']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'})
          )
plotDat0.sample(5)
plt.figure()
sns.lineplot(
    x = 'PAY_0',
    y = 'proportionDefaulted',
    data = plotDat0)


plotDat2 = (raw_df.groupby(['PAY_2']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'})
          )
plt.figure()
sns.lineplot(
    x = 'PAY_2',
    y = 'proportionDefaulted',
    data = plotDat2)

plotDat3 = (raw_df.groupby(['PAY_3']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'})
          )
plt.figure()
sns.lineplot(
    x = 'PAY_3',
    y = 'proportionDefaulted',
    data = plotDat3)

plotDat4 = (raw_df.groupby(['PAY_4']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'})
          )
plt.figure()
sns.lineplot(
    x = 'PAY_4',
    y = 'proportionDefaulted',
    data = plotDat4)

plotDat5 = (raw_df.groupby(['PAY_5']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'})
          )
plt.figure()
sns.lineplot(
    x = 'PAY_5',
    y = 'proportionDefaulted',
    data = plotDat5)

plotDat6 = (raw_df.groupby(['PAY_6']).apply(lambda x: len(x.default[x.default == 1])/len(x.default)).reset_index().rename(columns = {0: 'proportionDefaulted'})
            )
plt.figure()
sns.lineplot(
    x = 'PAY_6',
    y = 'proportionDefaulted',
    data = plotDat6)


# In[ ]:




