import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, RationalQuadratic, DotProduct
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
param_grid = {"alpha": np.linspace(1, 400, 10),
              "coef0": np.linspace(-5, 5, 5),
              "gamma": np.logspace(-3, 2, 10)}
krPoly = GridSearchCV(KernelRidge(kernel="poly", degree=2), cv=5, param_grid=param_grid)
krPoly.fit(X, y_poly)
y_testPoly = krPoly.predict(X_test)

plt.figure()
plt.plot(X, y_poly, 'r.', label='Data', markersize=7)
plt.plot(X_test, 0.1*X_test**2 - X_test, label='$f_p(x)$', linewidth=4)
plt.plot(X_test, y_testRBF, 'orange',label='KRR with RBF kernel', linewidth=4)
plt.plot(X_test, y_testPoly, label='KRR with polynomial kernel', linewidth=4)
plt.xlabel('$x$', fontsize=35)
plt.ylabel('$y$', fontsize=35)
plt.legend(fontsize=20)

#%%

# predict sine data with ESS
param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
krESS = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
krESS.fit(X, y_sin)
y_testESS = krESS.predict(X_test)

plt.figure()
plt.plot(X, y_sin, 'r.', label='Data', markersize=7)
plt.plot(X_test, np.sin(X_test), label='$f_s(x)$', linewidth=4)
plt.plot(X_test, y_testESS, label='KRR with ESS kernel', linewidth=4)
plt.xlabel('$x$', fontsize=35)
plt.ylabel('$y$', fontsize=35)
plt.legend(fontsize=20)

#%% GPR
gp_kernel = RBF() + WhiteKernel(0.1)
gpr_RBF = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10)
gpr_RBF.fit(X, y_poly)
y_testRBF, y_GPstd = gpr_RBF.predict(X_test, return_std=True)

gp_kernel = DotProduct(sigma_0=0)**2 + WhiteKernel(0.1)
gpr_poly = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=2)
gpr_poly.fit(X, y_poly)
y_testPoly, y_GPstd = gpr_poly.predict(X_test, return_std=True)

plt.figure()
plt.plot(X, y_poly, 'r.', label='Data', markersize=7)
plt.plot(X_test, 0.1*X_test**2 - X_test, label='$f_p(x)$', linewidth=4)
plt.plot(X_test, y_testRBF, 'orange', label='GPR with RBF kernel', linewidth=4)
plt.plot(X_test, y_testPoly, label='GPR with RQ kernel', linewidth=4)
plt.fill_between(X_test[:, 0], y_testPoly - y_GPstd, y_testPoly + y_GPstd, color='darkgreen',
                 alpha=0.2)
plt.xlabel('$x$', fontsize=35)
plt.ylabel('$y$', fontsize=35)
plt.legend(fontsize=20)

gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(0.1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=2)
gpr.fit(X, y_sin)
y_testESS, y_GPstd = gpr.predict(X_test, return_std=True)

plt.figure()
plt.plot(X, y_sin, 'r.', label='Data', markersize=7)
plt.plot(X_test, np.sin(X_test), label='$f_s(x)$', linewidth=4)
plt.plot(X_test, y_testESS, label='GPR with ESS kernel', linewidth=4)
plt.fill_between(X_test[:, 0], y_testESS - y_GPstd, y_testESS + y_GPstd, color='darkgreen',
                 alpha=0.2)
plt.xlabel('$x$', fontsize=35)
plt.ylabel('$y$', fontsize=35)
plt.legend(fontsize=20)
#%% SVR

# predict poly data with RBF 
param_grid = {"epsilon": np.linspace(0.05,0.15,9),
              "kernel": [RBF(l, p)
                         for l in np.linspace(0.5, 10, 10)
                         for p in np.logspace(0, 3, 10)]}
svRBF = GridSearchCV(SVR(C=0.5), cv=5, param_grid=param_grid)
svRBF.fit(X, y_poly)
y_testRBF = svRBF.predict(X_test)

# predict poly data with poly
param_grid = {"epsilon": np.linspace(0.05,0.15,9),
              "C": [1, 1e-1, 1e-2, 1e-3],
              "gamma": np.logspace(-2,1,9)}
svPoly = GridSearchCV(SVR(kernel="poly", degree=2), param_grid=param_grid)
svPoly.fit(X, y_poly)
y_testPoly = svPoly.predict(X_test)

plt.figure()
plt.plot(X, y_poly, 'r.', label='Data', markersize=7)
plt.plot(X_test, 0.1*X_test**2 - X_test, label='$f_p(x)$', linewidth=4)
plt.plot(X_test, y_testRBF, 'orange', label='SVR with RBF kernel', linewidth=4)
plt.plot(X_test, y_testPoly, label='SVR with polynomial kernel', linewidth=4)
plt.xlabel('$x$', fontsize=35)
plt.ylabel('$y$', fontsize=35)
plt.legend(fontsize=20)

#%%
param_grid = {"epsilon": np.linspace(0.15, 0.2, 5),
              "C": np.logspace(-1, 2, 9),
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
svESS = GridSearchCV(SVR(), param_grid=param_grid)
svESS.fit(X, y_sin) 
y_testESS = svESS.predict(X_test)

plt.figure()
plt.plot(X, y_sin, 'r.', label='Data', markersize=7)
plt.plot(X_test, np.sin(X_test), label='$f_s(x)$', linewidth=4)
plt.plot(X_test, y_testESS, label='SVR with ESS kernel', linewidth=4)
plt.xlabel('$x$', fontsize=35)
plt.ylabel('$y$', fontsize=35)
plt.legend(fontsize=20)

#%% Timing
import time
krESStest = KernelRidge(kernel=ExpSineSquared(4.64, 12.9), alpha=0.01)
t0 = time.time()
krESStest.fit(X, y_sin)
tKR = time.time() - t0

svESStest = SVR(kernel=ExpSineSquared(1.67,12.9), C=7.4989, epsilon=0.175)
t0 = time.time()
svESStest.fit(X, y_sin)
tSV = time.time() - t0
