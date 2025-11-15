import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# MLflow Tracking URI
#mlflow.set_tracking_uri("http://127.0.0.1:5000")   # Make sure MLflow server is running!


# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

max_depth = 2
n_estimators = 100   # not used by DecisionTree but kept for logging if needed

# Train decision tree model
rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#confusion_matrix
cm=confusion_matrix(y_pred,y_test)

#heatmap
sns.heatmap(cm,annot=True,cmap='viridis')
plt.savefig('cm.jpeg')

# Experiment Setup
mlflow.set_experiment('my_exp')
# Start MLflow run
with mlflow.start_run(run_name='rf_model'):
    # Log parameters
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimator',n_estimators)
    #mlflow.log_artifact('cm.jpeg')
    #mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf,'random_forest')
    mlflow.set_tag('author','jay')