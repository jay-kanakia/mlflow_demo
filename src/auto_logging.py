import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# MLflow Tracking URI
#mlflow.set_tracking_uri("http://ec2-13-203-78-91.ap-south-1.compute.amazonaws.com:5000/")   # Make sure MLflow server is running!

mlflow.set_experiment(experiment_name='autologging')
mlflow.autolog()

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Experiment Setup
#mlflow.set_experiment("iris_dt")

max_depth = 2
#n_estimators = 10   # not used by DecisionTree but kept for logging if needed

# Train decision tree model
dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(X_train, y_train)

# Evaluate the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

train_df=pd.DataFrame(X_train)
train_df['target']=y_train
train_df=mlflow.data.from_pandas(train_df)


test_df=pd.DataFrame(X_test)
test_df['target']=y_test
test_df=mlflow.data.from_pandas(test_df)

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='viridis')
plt.title("Confusion Matrix")
plt.savefig('cm.png')





# # Experiment Setup
# mlflow.set_experiment("Autologging")

# # Start MLflow run
# with mlflow.start_run(experiment_name='autolog1'):
#     # Log parameters
#     mlflow.log_metric('accuracy',accuracy)
#     mlflow.log_param('max_depth',max_depth)
    
#     mlflow.log_input(train_df,'training')
#     mlflow.log_input(test_df,'testing')