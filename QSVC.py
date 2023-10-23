import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile, execute, BasicAer, IBMQ
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_qulacs import QulacsProvider
simulator = QulacsProvider().get_backend()

#provider = IBMQ.load_account()

n_train =30
df = pd.read_csv('processed_train.csv')
df_test = pd.read_csv('processed_test.csv')
target = "risk_label_encoded"
X_train = np.ascontiguousarray(df.drop(target, axis=1).values)[:n_train]
y_train = np.array(df[target])[:n_train]
X_test = np.ascontiguousarray(df_test.drop(target, axis=1).values)
y_test = np.array(df_test[target])

# Encode the training and testing data onto a quantum circuit using the ZZFeatureMap
feature_map = ZZFeatureMap(X_train.shape[1], reps=2)
feature_map_ = ZZFeatureMap(X_test.shape[1], reps=2)

print('*'*100)
qkernel = QuantumKernel(feature_map ,quantum_instance=simulator)
qkernel_ = QuantumKernel(feature_map_ ,quantum_instance=simulator)
    # Set up the QSVC algorithm
print('instantiating qvc')
svm = QSVC(quantum_kernel=qkernel)
    # Train the QSVC algorithm on the training data
print('training')
svm.fit(X_train, y_train)
svm.fit(X_test, y_test)
print('done')
    # Test the QSVC algorithm on the testing data

svm_ = QSVC(quantum_kernel=qkernel_)
Ypred = svm_.predict(X_test)
print('done predict')
    # Print the accuracy of the QSVC algorithm
accuracy = np.mean(Ypred == y_test)
print("Accuracy:", accuracy)
    # Generate a classification report
report = classification_report(y_test, Ypred)

    # Print the classification report
print(report)
