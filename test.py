import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF
from sklearn.svm import SVR

n_samples = 100

rng = np.random.RandomState(0)

# Generate sample data
X = 15 * rng.rand(n_samples, 1)
y = np.sin(X).ravel()
y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise

param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [RBF(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
krRBF = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
krRBF.fit(X,y)

param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
krESS = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
krESS.fit(X,y)

gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(0.1)
gpr = GaussianProcessRegressor(kernel=gp_kernel)
gpr.fit(X, y)

#param_grid = {"C": np.linspace(0.5,1.5,9),
#              "epsilon": np.linspace(0.05,0.15,9),
#              "kernel": [ExpSineSquared(l, p)
#                         for l in np.logspace(-2, 2, 10)
#                         for p in np.logspace(0, 2, 10)]}
#clf = GridSearchCV(SVR(), cv=5, param_grid=param_grid)
#clf.fit(X, y) 

X_test = np.linspace(0,25,20000)[:,None]
y_testRBF = krRBF.predict(X_test)
y_testESS = krESS.predict(X_test)
y_testGP = gpr.predict(X_test)
#y_testSVR = clf.predict(X_test)

plt.figure()
plt.plot(X,y, 'r.', label='Data')
plt.plot(X_test, y_testRBF, label='KRR with RBF kernel')
plt.plot(X_test, y_testESS, label='KRR with ESS kernel')
plt.plot(X_test, y_testGP, label='GP with ESS + White kernels')
#plt.plot(X_test, y_testSVR, label='SVR')
plt.plot(X_test, np.sin(X_test), label='True relationship')
plt.legend()