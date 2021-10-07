# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 01:44:55 2021

@author: Albert
"""

import numpy as np
from scipy import linalg
from sklearn.metrics import r2_score, auc, roc_curve

class LR_ARD(object):
    def __init__(self):
        pass

    def fit(self, z, y, z_tst = None, y_tst = None,  hyper = None, prune = 1, maxit = 200, 
            pruning_crit = 1e-1, tol = 1e-6):
        self.z = z  #(NxK)
        self.z_tst = z_tst  #(NxK_tst)
        self.y_tst = y_tst  #(NxD_tst)
        self.y = y  #(NxD)
        self.K_tr = self.z @ self.z.T
        self.K_tst = self.z_tst @ self.z_tst.T
        
        self.K = self.z.shape[1] #num dimensiones input
        self.D = self.y.shape[1] #num dimensiones output
        self.N = self.z.shape[0] # num datos
        self.N_tst = self.z_tst.shape[0]
        
        # Some precomputed matrices
        #self.ZTZ = self.z.T @ self.z  #(KxK) es enorme, habría que ver si se puede evitar este calculo
        #self.YTZ = self.y.T @ self.z  #(DxK) 
        self.KTK = self.K_tr.T @ self.K_tr
        self.KTY = self.K_tr.T @ self.y
        self.YTK = self.y.T @ self.K_tr
        self.L = []
        self.mse = []
        self.mse_tst = []        
        self.R2 = []
        self.R2_tst = []
        #self.AUC = []
        #self.AUC_tst = []
        self.K_vec = []
        self.labels_pred = []
        self.input_idx = np.ones(self.K, bool)
        if hyper == None:
            self.hyper = HyperParameters(self.K)
        else:
            self.hyper = hyper
        self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper)

        self.fit_vb(prune, maxit, pruning_crit, tol)

    def pruning(self, pruning_crit):       
        q = self.q_dist
        
        fact_sel = np.array([])
        for K in np.arange(q.K):
            if any(abs((self.z.T @ q.A['mean'])[:,K]) > pruning_crit):
                fact_sel = np.append(fact_sel,K)
        fact_sel = np.unique(fact_sel).astype(int)
        
        aux = self.input_idx[self.input_idx]
        aux[fact_sel] = False
        self.input_idx[self.input_idx] = ~aux
        
        # Pruning W and alpha
#        q.W['mean'] = q.W['mean'][:,fact_sel]
#        q.W['cov'] = q.W['cov'][fact_sel,:][:,fact_sel]
#        q.W['prodT'] = q.W['prodT'][fact_sel,:][:,fact_sel]
        q.A['mean'] = q.A['mean'][:,fact_sel]
        q.A['cov'] = q.A['cov'][fact_sel,:][:,fact_sel]
        q.A['prodT'] = q.A['prodT'][fact_sel,:][:,fact_sel]
        self.z = self.z[:,fact_sel]
        self.z_tst = self.z_tst[:,fact_sel]
        self.ZTZ = self.ZTZ[fact_sel,:][:,fact_sel]
        self.YTZ = self.YTZ[:,fact_sel]
        q.alpha['a'] = q.alpha['a'][fact_sel]
        q.alpha['b'] = q.alpha['b'][fact_sel]
        self.hyper.alpha_a = self.hyper.alpha_a[fact_sel]
        self.hyper.alpha_b = self.hyper.alpha_b[fact_sel]
        q.K = len(fact_sel)
        
    def compute_mse(self, z = None, y = None):
        q = self.q_dist
        if z is None:
            z = self.z
        if y is None:
            y = self.y
        diff = (y - z @ self.z.T @ q.A['mean']).ravel()
        return  diff@diff/self.N
    
    def compute_R2(self, z = None, y = None):
        q = self.q_dist
        if z is None:
            z = self.z
        if y is None:
            y = self.y
        return  r2_score(y.ravel(), (z @ self.z.T @ q.A['mean']).ravel())
#    def compute_AUC(self, z = None, y = None):
#        q = self.q_dist
#        if z is None:
#            z = self.z
#        if y is None:
#            y = self.y
#        fpr, tpr, thresholds = roc_curve(y.ravel(), (z @ self.z.T @ q.A['mean']).ravel())
#        return auc(fpr, tpr)
        
    def predict(self, Z_test):
        q = self.q_dist
        return Z_test[:,self.input_idx] @ self.z.T @ q.A['mean']
        
    def fit_vb(self, prune, maxit=200, pruning_crit = 1e-1, tol = 1e-6):
        q = self.q_dist
        for i in range(maxit):
            self.update()
            self.mse.append(self.compute_mse())
            self.R2.append(self.compute_R2())
            #self.AUC.append(self.compute_AUC())
            self.mse_tst.append(self.compute_mse(self.z_tst, self.y_tst))
            self.R2_tst.append(self.compute_R2(self.z_tst, self.y_tst))
            #self.AUC_tst.append(self.compute_AUC(self.z_tst, self.y_tst))
            #print(self.mse_tst[-1])
            self.K_vec.append(q.K)
            if self.predict(self.z_tst) > 0.5:
                self.labels_pred.append(1)
            else:
                self.labels_pred.append(0)
            #print(self.predict(self.z_tst))
            # self.depruning(1e-15)
            self.L.append(self.update_bound())
            if prune:
                self.pruning(pruning_crit)
            if q.K == 0:
                print('\nThere are no representative latent factors, no structure found in the data.')
                return
            print('\rIteration %d Lower Bound %.1f K %4d' %(i+1, self.L[-1], q.K), end='\r', flush=True)
            if (len(self.L) > 100) and (abs(1 - np.mean(self.L[-101:-1])/self.L[-1]) < tol):
                print('\nModel correctly trained. Convergence achieved')
                return                
        print('')

    def update(self):
        self.update_a()
        #self.update_w()
        self.update_alpha()
        self.update_tau()

    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:
            L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) #np.linalg.cholesky(A)
            return np.dot(L.T,L) #linalg.pinv(L)*linalg.pinv(L.T)
        except:
            return np.nan
        
#    def update_w(self):
#        q = self.q_dist
#        # cov
#        w_cov = self.myInverse(np.diag(q.alpha_mean()) + q.tau_mean() * self.ZTZ)
#        # Efficient and robust way of computing:  solve(diag(alpha) + tau * Z^TZ)
#        # tmp = 1/np.sqrt(q.alpha_mean())
#        # aux = np.outer(tmp,tmp)*self.ZTZ + np.eye(q.K)/q.tau_mean()
#        # cho = np.linalg.cholesky(aux)            
#        # w_cov = 1/q.tau_mean() * np.outer(tmp,tmp) * np.dot(linalg.pinv(cho.T),linalg.pinv(cho))
#        
#        if not np.any(np.isnan(w_cov)):
#            q.W['cov'] = w_cov
#            # mean
#            q.W['mean'] = q.tau_mean() * self.YTZ @ q.W['cov']
#            #E[W*W^T]
#            q.W['prodT'] = q.W['mean'].T @ q.W['mean']+self.D*q.W['cov']
#            #q.W['prodT_diag'] =  q.W['mean']**2 + self.D*np.diag(q.W['cov'])
#           
#        else:
#            print ('Cov W is not invertible, not updated')
    def update_a(self):
        q = self.q_dist
        # cov
        
        ##############
        #a_cov = self.myInverse(self.z @ np.diag(q.alpha_mean()) @ self.z.T + q.tau_mean() * self.KTK)
        ##############
        a_cov = self.myInverse(np.multiply(q.alpha_mean(), self.z) @ self.z.T + q.tau_mean() * self.KTK)
        
        # Efficient and robust way of computing:  solve(diag(alpha) + tau * Z^TZ)
        # tmp = 1/np.sqrt(q.alpha_mean())
        # aux = np.outer(tmp,tmp)*self.ZTZ + np.eye(q.K)/q.tau_mean()
        # cho = np.linalg.cholesky(aux)            
        # w_cov = 1/q.tau_mean() * np.outer(tmp,tmp) * np.dot(linalg.pinv(cho.T),linalg.pinv(cho))
        
        if not np.any(np.isnan(a_cov)):
            q.A['cov'] = a_cov
            
            # mean
            q.A['mean'] = q.tau_mean() *q.A['cov'] @ self.KTY 
            #E[W*W^T]
            #q.A['prodT'] = q.A['mean'].T @ self.K_tr @ q.A['mean']+self.D*q.A['cov']
            #q.A['prodT'] = q.A['mean'].T @ q.A['mean']+self.D*q.A['cov']
            q.A['prodT'] = q.A['mean']**2 +self.D*q.A['cov']
            #q.W['prodT_diag'] =  q.W['mean']**2 + self.D*np.diag(q.W['cov'])
           
        else:
            print ('Cov A is not invertible, not updated')

      
    def update_alpha(self):
        q = self.q_dist
        q.alpha['a'] = (self.hyper.alpha_a + 0.5 * self.D)/(self.D)
        #prod = np.diag(self.z.T @ q.A['prodT'] @ self.z)
        prod = np.multiply(self.z.T @ q.A['prodT'], self.z.T).sum(-1)
        q.alpha['b'] = (self.hyper.alpha_b + 0.5 * prod)/(self.D)
        
    def update_tau(self):
        q = self.q_dist
        q.tau['a'] = (self.hyper.tau_a + 0.5 * self.D*self.N)/(self.D*self.N) 
        q.tau['b'] = (self.hyper.tau_b + 0.5 *(np.sum(self.y.ravel()**2)+ np.trace(self.K_tr.T @ self.K_tr @ q.A['prodT']) - 2 * np.trace(self.YTK @ q.A['mean'])))/(self.D*self.N)   
        
    
    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
        
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H
        
    def update_bound(self):
        """Update the Lower Bound.
        
        Uses the learnt variables of the model to update the lower bound.
        
        """
        
        q = self.q_dist
        
        q.A['LH'] = self.HGauss(q.A['mean'], q.A['cov'], q.A['LH'])
        #q.W['LH'] = self.HGauss(q.W['mean'], q.W['cov'], q.W['LH'])
        #self.W['LH'] = self.z.T @ q.A['LH']
        # Entropy of alpha and tau
        # q.alpha['LH'] = np.sum(self.HGamma(q.alpha['a'], q.alpha['b']))
        # q.tau['LH'] = np.sum(self.HGamma(q.tau['a'], q.tau['b']))
            
        # Total entropy
        # EntropyQ = q.W['LH'] + q.alpha['LH']  + q.tau['LH']
           
        # Calculation of the E[log(p(Theta))]
        q.tau['ElogpXtau'] = -(0.5 *  self.N * self.D + self.hyper.tau_a - 2)* np.log(q.tau['b'])
        q.alpha['ElogpWalp'] = -(0.5 * self.D + np.mean(self.hyper.alpha_a) - 2)* np.sum(np.log(q.alpha['b']))
        
        # Total E[log(p(Theta))]
        ElogP = q.tau['ElogpXtau'] + q.alpha['ElogpWalp']
        return ElogP - q.A['LH']        

class HyperParameters(object):
    def __init__(self, K):
        self.alpha_a = 2 * np.ones((K,))
        self.alpha_b = 1 * np.ones((K,))
        self.tau_a = 1e-14
        self.tau_b = 1e-14

class Qdistribution(object):
    def __init__(self, n, D, K, hyper):
        self.n = n
        self.D = D
        self.K = K
        
        # Initialize gamma disributions
        alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.K)
        self.alpha = alpha 
        tau = self.qGamma(hyper.tau_a,hyper.tau_b,1)
        self.tau = tau 

        # The remaning parameters at random
        self.init_rnd()

    def init_rnd(self):
        self.A = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
            
#        self.W["mean"] = np.random.normal(0.0, 1.0, self.D * self.K).reshape(self.D, self.K)
#        self.W["cov"] = np.eye(self.K)
#        self.W["prodT"] = np.dot(self.W["mean"].T, self.W["mean"])+self.K*self.W["cov"]
        
        
        
        #self.A["mean"] = np.random.normal(0.0, 1.0, self.D * self.K).reshape(self.D, self.K)
        self.A["mean"] = np.random.normal(0.0, 1.0, self.n).reshape(self.n,1)
        #self.A["cov"] = np.eye(self.K)
        self.A["cov"] = np.eye(self.n)
        #self.A["prodT"] = np.dot(self.A["mean"].T, self.A["mean"])+self.K*self.A["cov"]
        self.A["prodT"] = np.dot(self.A["mean"].T, self.A["mean"])+self.n*self.A["cov"]
        
        

    def qGamma(self,a,b,K):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [1, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [K, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __K: array (shape = [K, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = {                
                "a":         a,
                "b":         b,
                "LH":         None,
                "ElogpWalp":  None,
            }
        return param
        
    def alpha_mean(self):
        return self.alpha['a'] / self.alpha['b']
    
    def tau_mean(self):
        return self.tau['a'] / self.tau['b']
    
