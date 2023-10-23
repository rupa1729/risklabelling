import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import roc_auc_score
import time 

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



#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)


# In[29]:


# svc = SVC(kernel='poly', degree=5, random_state=seed)
svc = SVC(probability=True, random_state=seed)
svc.fit(X_train, y_train)
svc.score(X_train, y_train)


# In[30]:


#print("Train")
print(classification_report(y_train, svc.predict(X_train)))
#print("Test")
print(classification_report(y_test, svc.predict(X_test)))


# In[31]:


print("Train AUC:", roc_auc_score(y_train, svc.predict_proba(X_train), multi_class='ovr'))
print("Test AUC:", roc_auc_score(y_test, svc.predict_proba(X_test), multi_class='ovr'))
