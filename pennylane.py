import numpy as np  # linear algebra
import pandas as pd  
import jax
import pennylane as qml
import optax

from pennylane import numpy as pnp
from jax_utils import square_kernel_matrix_jax, kernel_matrix_jax, target_alignment_jax

seed = 42
np.random.seed(seed)

df = pd.read_csv('processed_train.csv')[:20]   # load only the first 20 data points for the training dataset
df_test = pd.read_csv('processed_test.csv')[:1000]  # load only the first 1000 data points for the testing dataset

# The target variable
target = "risk_label_encoded"

# Prepare the training data as numpy arrays
X_train = np.ascontiguousarray(df.drop(target, axis=1).values)
y_train = np.array(df[target])

# Prepare the testing data as numpy arrays
X_test = np.ascontiguousarray(df_test.drop(target, axis=1).values)
y_test = np.array(df_test[target])



# In[54]:


def feature_map(x, params, n_layers, n_wires):
    """The embedding ansatz"""
    steps = x.shape[0]//3
    qubits = list(range(n_wires))
    
    for q in qubits:
        qml.Hadamard(wires=q)
    
    for l in range(n_layers):
        for q in qubits:
            for i in range(steps):
                z = x[3*i:3*i+3]*params[l,q,0,3*i:3*i+3] + params[l,q,1,3*i:3*i+3]
                qml.Rot(z[0], z[1], z[2], wires=q)
                
        # Entanglement layer
        for i in range(n_wires - 1):
            qml.CZ((i, i + 1))


# In[62]:


n_l = 2
n_w = 4
in_shape = 6

dev = qml.device("default.qubit.jax", wires=n_w)
params_shape = (n_l,n_w,2,in_shape)
params = pnp.random.uniform(0, 2 * np.pi, params_shape, requires_grad=True)


# In[63]:


@qml.qnode(dev, interface = 'jax')
def kernel_circuit(x1, x2, params):
    feature_map(x1, params, n_l, n_w)
    qml.adjoint(feature_map)(x2, params, n_l, n_w)
    return qml.probs(wires=range(n_w))

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]


# In[64]:


print(qml.draw(kernel_circuit)(X_train[0], X_train[1], params))


# In[65]:


jit_kernel = jax.jit(kernel)


# In[66]:


start = time.time()

init_kernel = lambda x1, x2: jit_kernel(x1, x2, params)
kernel_matrix = lambda X1, X2: kernel_matrix_jax(X1, X2, init_kernel)

qsvc = SVC(probability=True, kernel=kernel_matrix, random_state=seed)
qsvc.fit(X_train, y_train)

end = time.time()
print("Duration:", end - start)


# In[67]:


print("Train")
print(classification_report(y_train, qsvc.predict(X_train)))
print("Test")
print(classification_report(y_test, qsvc.predict(X_test)))


# In[68]:


print("Train AUC:", roc_auc_score(y_train, qsvc.predict_proba(X_train), multi_class='ovr'))
print("Test AUC:", roc_auc_score(y_test, qsvc.predict_proba(X_test), multi_class='ovr'))