import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
seed = 42
np.random.seed(seed)

#X_train = X_smote[features].to_numpy()
#y_train = y_smote.to_numpy()

#X_test = X_test_scaled[features].to_numpy()
#y_test = y_test_raw.to_numpy()


# Loading the datasets
df = pd.read_csv('processed_train.csv')[:30]   # load only the first 20 data points for the training dataset
df_test = pd.read_csv('processed_test.csv')[:1000]  # load only the first 1000 data points for the testing dataset

# The target variable
target = "risk_label_encoded"

# Prepare the training data as numpy arrays
X_train = np.ascontiguousarray(df.drop(target, axis=1).values)
y_train = np.array(df[target])

# Prepare the testing data as numpy arrays
X_test = np.ascontiguousarray(df_test.drop(target, axis=1).values)
y_test = np.array(df_test[target])



# instantiate the classifier
rfc = RandomForestClassifier(random_state=seed,
                             n_jobs=-1,
                             n_estimators=46,
                             class_weight="balanced_subsample",
                             bootstrap=True,
                             max_depth=15,
                             min_samples_leaf=1,
                             min_samples_split=2)

_ = rfc.fit(X_train, y_train)
print('Training set score: {:.4f}'.format(rfc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(rfc.score(X_test, y_test)))

from sklearn.metrics import classification_report

print("Train")
print(classification_report(y_train,rfc.predict(X_train)))
print()
print("Test")
print(classification_report(y_test,rfc.predict(X_test)))

print("Train AUC:", roc_auc_score(y_train, rfc.predict_proba(X_train), multi_class='ovr'))
print("Test AUC:", roc_auc_score(y_test, rfc.predict_proba(X_test), multi_class='ovr'))
