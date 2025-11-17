import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

#mlflow.set_tracking_uri("http://127.0.0.1:5000")

#loading data
df=pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#splitting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#scaling data
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#creating dataframe
train_df=pd.DataFrame(X_train,columns=X.columns)
test_df=pd.DataFrame(X_test,columns=X.columns)

train_df['outcome']=y_train
test_df['outcome']=y_test

#building model
rf=RandomForestClassifier()
rf.fit(X_train,y_train)

grid=({
    'n_estimators':[50,150,250],
    'max_depth' :[10,5,2]
    
})

# grid search cv
gsc=GridSearchCV(estimator=rf,param_grid=grid,scoring='accuracy',cv=5)
gsc.fit(X_train,y_train)
best_param=gsc.best_params_
best_score=gsc.best_score_




mlflow.set_experiment(experiment_name='nested_exp')
with mlflow.start_run(run_name='nested_run1') as parent:
   for i in range(len(gsc.cv_results_['params'])):
      with mlflow.start_run(nested=True) as child:
         mlflow.log_params(gsc.cv_results_['params'][i])
         mlflow.log_metric("mean test score",gsc.cv_results_['mean_test_score'][i])