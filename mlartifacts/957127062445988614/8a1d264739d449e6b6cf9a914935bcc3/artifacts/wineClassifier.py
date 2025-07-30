import yaml
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

mlflow.set_tracking_uri("http://127.0.0.1:5000")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

rf_params = params["random_forest"]
max_depth = rf_params["max_depth"]
n_estimators = rf_params["n_estimators"]
test_size = rf_params["test_size"]
random_state = rf_params["random_state"]

wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

mlflow.set_experiment('banana-pudding')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('test_size', test_size)
    mlflow.log_param('random_state', random_state)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("Confusion-matrix.png")

    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({"Author": "Pawan", "Project": "Wine Classification"})
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print("Accuracy: ", accuracy)