import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, RationalQuadratic
from sklearn.svm import SVR
#%%
n_samples = 100

rng = np.random.RandomState(0)

# Generate sample data
X = 15 * rng.rand(n_samples, 1)
X_test = np.linspace(0,25,1000)[:,None]

y_sin = np.sin(X).ravel()
y_sin += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise

y_poly = (0.1*X**2 - X).ravel()
y_poly += 10 * (0.5 - rng.rand(X.shape[0]))

y_comb = (X*np.sin(X)).ravel()
y_comb += 3 * (0.5 - rng.rand(X.shape[0]))
#%%  
# predict poly data with RBF 
param_grid = {"alpha": [1e-1, 1e-2, 1e-3],
              "kernel": [RBF(l, p)
                         for l in np.logspace(-3, 3, 10)
                         for p in np.logspace(0, 3, 10)]}
krRBF = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
krRBF.fit(X, y_poly)
y_testRBF = krRBF.predict(X_test)

# predict poly data with poly
param_grid = {"alpha": np.logspace(-3, 1, 10)}
krPoly = GridSearchCV(KernelRidge(kernel="poly", degree=2), cv=5, param_grid=param_grid)
krPoly.fit(X, y_poly)
y_testPoly = krPoly.predict(X_test)

plt.figure()
plt.plot(X, y_poly, 'r.', label='Data')
plt.plot(X_test, 0.1*X_test**2 - X_test, label='True relationship')
plt.plot(X_test, y_testRBF, label='KRR with RBF kernel')
plt.plot(X_test, y_testPoly, label='KRR with polynomial kernel')
plt.legend()

# predict sine data with ESS
param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
krESS = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
krESS.fit(X, y_sin)
y_testESS = krESS.predict(X_test)

plt.figure()
plt.plot(X, y_sin, 'r.', label='Data')
plt.plot(X_test, np.sin(X_test), label='True relationship')
plt.plot(X_test, y_testESS, label='KRR with ESS kernel')
plt.legend()

#%% GPR
gp_kernel = RBF() + WhiteKernel(0.1)
gpr_RBF = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=2)
gpr_RBF.fit(X, y_poly)
y_testRBF, y_GPstd = gpr_RBF.predict(X_test, return_std=True)

gp_kernel = RationalQuadratic() + WhiteKernel(0.1)
gpr_poly = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=2)
gpr_poly.fit(X, y_poly)
y_testPoly, y_GPstd = gpr_poly.predict(X_test, return_std=True)

plt.figure()
plt.plot(X, y_poly, 'r.', label='Data')
plt.plot(X_test, 0.1*X_test**2 - X_test, label='True relationship')
plt.plot(X_test, y_testRBF, label='GP with RBF kernel')
plt.plot(X_test, y_testPoly, label='GP with RQ kernel')
plt.fill_between(X_test[:, 0], y_testPoly - y_GPstd, y_testPoly + y_GPstd, color='darkorange',
                 alpha=0.2)
plt.legend()

gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(0.1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=2)
gpr.fit(X, y_sin)
y_testESS, y_GPstd = gpr.predict(X_test, return_std=True)

plt.figure()
plt.plot(X, y_sin, 'r.', label='Data')
plt.plot(X_test, np.sin(X_test), label='True relationship')
plt.plot(X_test, y_testESS, label='GPR with ESS kernel')
plt.fill_between(X_test[:, 0], y_testESS - y_GPstd, y_testESS + y_GPstd, color='darkorange',
                 alpha=0.2)
plt.legend()
#%% SVR

# predict poly data with RBF 
param_grid = {"epsilon": np.linspace(0.05,0.15,9),
              "kernel": [RBF(l, p)
                         for l in np.logspace(-3, 3, 10)
                         for p in np.logspace(0, 3, 10)]}
svRBF = GridSearchCV(SVR(C=0.5), cv=5, param_grid=param_grid)
svRBF.fit(X, y_poly)
y_testRBF = svRBF.predict(X_test)

# predict poly data with poly
param_grid = {"C": np.logspace(-1,2,10)}
svPoly = SVR(kernel="poly", degree=2, C=0.5)
svPoly.fit(X, y_poly)
y_testPoly = svPoly.predict(X_test)

plt.figure()
plt.plot(X, y_poly, 'r.', label='Data')
plt.plot(X_test, 0.1*X_test**2 - X_test, label='True relationship')
plt.plot(X_test, y_testRBF, label='SVR with RBF kernel')
plt.plot(X_test, y_testPoly, label='SVR with polynomial kernel')
plt.legend()

param_grid = {"epsilon": np.linspace(0.15, 0.2, 5),
              "C": np.logspace(-1, 2, 9),
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
sv = GridSearchCV(SVR(), param_grid=param_grid)
sv.fit(X, y_sin) 
y_testESS = sv.predict(X_test)

plt.figure()
plt.plot(X, y_sin, 'r.', label='Data')
plt.plot(X_test, np.sin(X_test), label='True relationship')
plt.plot(X_test, y_testESS, label='SVR with ESS kernel')
plt.legend()
