#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script implementing the code described in Nocquet (2018).

==============================================================================

It is a fast fault slip inversion code with non-negativity or bound constraints 
imposed as a prior.
Maximum a posteriori probability, joint posterior PDF and marginals PDF can be
derived independently.

WARNING : joint posterior PDF and marginals PDF computations do not work in
complex cases with between 50 and 100 parameters. 


Author : Roxane Ferry
First version, July 2020.
"""

# Imports
import numpy as np
import scipy
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
from scipy.integrate import quad
import scipy.stats
import matplotlib.pyplot as plt

G = np.array([[-7., -4.], [1., 10.], [2., -11.]])
d = np.array([[10.], [3.], [-5.]])
Cd = np.identity(3, dtype=float) * 5.**2
Cm = np.identity(2, dtype=float) * (0.5 / 2)**2
Cd_inv = np.linalg.inv(Cd)
Cm_inv = np.linalg.inv(Cm)
m0 = np.array([[0.5], [0.5]])
m = np.array([[0.5], [0.3]])

def MAP(G, d, Cd, Cm, m0, method="non-neg", bounds=(0, 10)):
    """
    Maximum a posteriori probability estimate. 
    Solves Eq.15 from Nocquet (2018).

    Parameters
    ----------
    G : array, shape (n, p)
        Model matrix.
    d : array, shape (n,)
        Data vector.
    Cd : array, shape (p, p)
        Data covariance matrix.
    Cm : array, shape (n, n)
        Model covariance matrix.
    m0 : array, shape (p,)
        Mean of a priori Gaussian density probability function.
    method : "non-neg" or "bounded", optional
        Method to compute least-squares. 
            * "non-neg" : for the non-negative case,  uses scipy.optimize.nnls 
              based on the non-negative least-squares algorithm from 
              Lawson & Hanson (1974).
            * "bounded" : for the bounded case, uses scipy.optimize.lsq_linear
              based on the Bounded-Variable Least-Squares algorithm from 
              Stark & Parker (1995).
        The default is "non-neg".
    bounds : 2-tuple or array-like, optional
        Lower and upper bounds on independent variables. Each array must have 
        shape (p,) or be a scalar, in the latter case a bound will be the same
        for all variables. The default is (0, 10).


    Returns
    -------
    array, shape (p, )
        Solution found.

    """
    
    # Preliminary checks
    if method not in ["non-neg", "bounded"]:
        raise ValueError('"method" must be "non-neg" or "bvls"')
    
    
    # Squared root of Cm and Cd
    Cm_sqrt = scipy.linalg.sqrtm(Cm)
    Cd_sqrt = scipy.linalg.sqrtm(Cd)
    
    # Inverse of the squared root of Cm and Cd
    iCm_sqrt = np.linalg.inv(Cm_sqrt)
    iCd_sqrt = np.linalg.inv(Cd_sqrt)
    
    A = np.concatenate((iCd_sqrt @ G, iCm_sqrt))
    B = np.concatenate((iCd_sqrt @ d, iCm_sqrt @ m0))
    
    # Select correct inverse method
    if method=="non-neg":
        inv = nnls(A, B)
        res = inv[0]
    elif method=="bounded":
        inv = lsq_linear(A, B, bounds=bounds, method="bvls", verbose=2)
        res = inv.x

    return res


def joint_posterior_pdf(G, d, Cd, Cm, m, m0, bounds=(0, 10)):
    """
    Joint posterior probability of the set of parameters m.
    Solves Eq.8 or 9 from Nocquet (2018).
    Uses TMVN approximation.

    Parameters
    ----------
    G : array, shape (n, p)
        Model matrix.
    d : array, shape (n,)
        Data vector.
    Cd : array, shape (p, p)
        Data covariance matrix.
    Cm : array, shape (n, n)
        Model covariance matrix.
    m : array, shape (p,)
        Parameters values to be evaluated 
    m0 : array, shape (p,)
        Mean of a priori Gaussian density probability function.
    bounds : 2-tuple or array-like, optional
        Lower and upper bounds on independent variables. Each array must have 
        shape (p,) or be a scalar, in the latter case a bound will be the same
        for all variables. The default is (0, 10).


    Returns
    -------
    float
        Joint posterior probability value for m values.
    """
    
    # Inverse of Cm and Cd
    iCm = np.linalg.inv(Cm)
    iCd = np.linalg.inv(Cd)
    
    # Computation of Cmtilde and mtilde
    Cmtilde = np.linalg.inv((G.T @ iCd @ G) + iCm)
    mtilde = np.linalg.inv((G.T @ iCd @ G) + iCm) @ ((G.T @ iCd @ d) + 
                                                     iCm @ m0)
    
    # DEALING WITH BOUNDS
    # Function to convert tuple bounds to array bounds 
    def prepare_bounds(bounds, p):
        lb, ub = [np.asarray(b, dtype=float) for b in bounds]
    
        if lb.ndim == 0:
            lb = np.resize(lb, p)
    
        if ub.ndim == 0:
            ub = np.resize(ub, p)
    
        return lb, ub    

    # Parameters number
    p = m0.shape[0] 
    
    # Computes lower and upper bounds
    lb, ub = prepare_bounds(bounds, p)    
    
    # Checks that bounds are correctly defined
    if lb.shape != (p,) and ub.shape != (p,):
        raise ValueError("Bounds have wrong shape.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")
    
    
    Kplus, error = scipy.stats.mvn.mvnun(lb, ub, mtilde, Cmtilde)
    assert error==0, 'Potential errors in TMVN approximation'
    Kplus = 1./Kplus
    sigma = Kplus * np.exp(-0.5 * ((m - mtilde).T @ np.linalg.inv(Cmtilde) @ (m - mtilde)))
    
    return sigma


def likelihood(G, d, Cd, m):
    Cd_inv = np.linalg.inv(Cd)
    return 0.5 * (((G @ m) - d).transpose() @ Cd_inv @ ((G @ m) - d))


def exact_uniform(G, d, Cd, m, m0):
    """
    Exact solution for the uniform overdetermined case (Eq 22 in Nocquet (2018)).
    """
    
    iCd = np.linalg.inv(Cd)
    detCd = np.linalg.det(Cd)
    n = len(d)
    
    Cmtilde = np.linalg.inv(G.T @ iCd @ G)  # Eq 22
    mtilde = Cmtilde @ G.T @ iCd @ d  # Eq 22
    
    Kd = 1. / ((2 * np.pi)**(1/n) * np.sqrt(detCd))
    K0 = d.T @ iCd @ (d - G @ mtilde)  # Eq B2
    Ku0 = np.exp(-0.5 * K0) * Kd
    
    jppdf = Ku0 * np.exp(-0.5 * ((m - mtilde).T @ np.linalg.inv(Cmtilde) @ (m - mtilde)))
    
    return jppdf


def multi_gauss_distrib(G, d, Cd, Cm, m, m0):
    iCd = np.linalg.inv(Cd)
    iCm = np.linalg.inv(Cm)   
    n= len(d)
    
    Cmtilde = np.linalg.inv((G.T @ iCd @ G) + iCm)
    mtilde = np.linalg.inv((G.T @ iCd @ G) + iCm) @ ((G.T @ iCd @ d) + iCm @ m0)
    
    iCmtilde = np.linalg.inv(Cmtilde)
    detCmtilde = np.linalg.det(Cmtilde)
    
    res = (2 * np.pi)**(-n/2) * detCmtilde**(0.5)  * np.exp(-0.5 * ((m - mtilde).T @ iCmtilde @ (m - mtilde)))
    return res
    

def marginal_pdf(G, d, Cd, Cm, m0, m_value, ind_m1, bounds=(0, 1)):
    """
    ind_m1 : index in m of the value to evaluate 
    m_value : value of m to evaluate
    
    bounds are constant for all m
    """
    # # Computes inverse
    # iCd = np.linalg.inv(Cd)
    # iCm = np.linalg.inv(Cm)
    
    # # Index of the last element 
    # p = np.shape(Cm)[0] - 1 
    
    # # Reorganizes m0
    # rm0 = np.concatenate((m0[ind_m1], np.delete(m0, ind_m1)))
    # rm0 = rm0.reshape(-1, 1) # to be a column vector
    
    # # Computes Cmtilde and mtilde
    # Cmtilde = np.linalg.inv((G.T @ iCd @ G) + iCm)    
    # mtilde = np.linalg.inv((G.T @ iCd @ G) + iCm) @ ((G.T @ iCd @ d) + iCm @ rm0)

    # # Reorganise Cmtilde
    # rCmtilde = np.copy(Cmtilde)  # initialisation
    
    # if ind_m1 == 0:  # if it is the first value, no need to reorganise
    #     pass
    
    # elif ind_m1 == p :  # if it is the last value of m
    #     rCmtilde[1:ind_m1, 0] = np.copy(Cmtilde[1:ind_m1, ind_m1])
    #     rCmtilde[1:ind_m1, ind_m1] = np.copy(Cmtilde[1:ind_m1, 0])
        
    #     rCmtilde[0, 1:-1] = np.copy(Cmtilde[ind_m1, 1:-1])
    #     rCmtilde[ind_m1, 1:-1] = np.copy(Cmtilde[0, 1:-1])
    
    # elif ind_m1 == 1 :  # if it is the second one
    #     rCmtilde[1+ind_m1, 0] = np.copy(Cmtilde[1+ind_m1, ind_m1])
    #     rCmtilde[1+ind_m1, ind_m1] = np.copy(Cmtilde[1+ind_m1, 0])

    #     rCmtilde[0, 2:] = np.copy(Cmtilde[ind_m1, 2:])
    #     rCmtilde[ind_m1, 2:] = np.copy(Cmtilde[0, 2:])
        
    # else :
    #     rCmtilde[1:ind_m1, 0] = np.copy(Cmtilde[1:ind_m1, ind_m1])
    #     rCmtilde[1:ind_m1, ind_m1] = np.copy(Cmtilde[1:ind_m1, 0])
        
    #     rCmtilde[ind_m1+1, 0] = np.copy(Cmtilde[ind_m1+1, ind_m1])
    #     rCmtilde[ind_m1+1, ind_m1] = np.copy(Cmtilde[ind_m1+1, 0])
        
    #     rCmtilde[0, 1:ind_m1] = np.copy(Cmtilde[ind_m1, 1:ind_m1])
    #     rCmtilde[ind_m1, 1:ind_m1] = np.copy(Cmtilde[0, 1:ind_m1])
        
    #     rCmtilde[0, ind_m1+1:] = np.copy(Cmtilde[ind_m1, ind_m1+1:])
    #     rCmtilde[ind_m1, ind_m1+1:] = np.copy(Cmtilde[0, ind_m1+1:])
     
    # # Switch diagonal values
    # rCmtilde[0,0] = np.copy(Cmtilde[ind_m1, ind_m1])
    # rCmtilde[ind_m1, ind_m1] = np.copy(Cmtilde[0,0])
    
    # # Partitioned Cmtilde
    # Cmtilde11 = rCmtilde[0,0]ind_m1 = 1
    # Cmtilde21 = rCmtilde[0, 1:]
    # Cmtilde12 = rCmtilde[1:, 0]
    # Cmtilde22 = rCmtilde[1:, 1:]
    
    # # Inverse of Cmtilde11
    # if isinstance(Cmtilde11, (float, int)):  # if Cmtilde11 is not an array        
    #     iCmtilde11 = np.array([1. / Cmtilde11])
    # else :
    #     iCmtilde11 = np.linalg.inv(Cmtilde11)
    
    # # Reorganise mtilde
    # rmtilde = np.copy(mtilde)
    # rmtilde[0] = mtilde[ind_m1]
    # try:
    #     rmtilde[1:] = np.delete(mtilde, ind_m1).reshape(np.shape(rmtilde[1:]))
    # except ValueError:
    #     rmtilde[1:] = np.delete(mtilde, ind_m1)
    
    # # Partitioned rmtilde
    # rmtilde1 = rmtilde[0]
    # rmtilde2 = rmtilde[1:] 

    # # Computes A, iA and defines b(x)
    # # if isinstance(Cmtilde12.T @ iCmtilde11, (float, int, np.float64, np.int64, np.float32, np.int32)):
    # if (len(Cmtilde) == 1 and len(iCmtilde11) ==1):
    # # If Cmtilde12 and Cmtilde11 are arrays of one element
    #     A = Cmtilde22 - (np.array([Cmtilde12.T @ iCmtilde11]) @ Cmtilde12)
    #     iA = 1. / A
    
    #     def b(x):
    #         return rmtilde2 + (np.array([Cmtilde12.T @ iCmtilde11]) @ (x - rmtilde1))
    # else :
    # # If Cmtilde12 and Cmtilde11 are not arrays
    #     A = Cmtilde22 - (Cmtilde12.T @ iCmtilde11 @ Cmtilde12)
    #     # Cmtilde12.reshape(-1, 1) is equivalent to Cmtilde12.T
    #     iA = np.linalg.inv(A)
    #     def b(x):
    #             return rmtilde2 + (Cmtilde12.T @ iCmtilde11 @ (x - rmtilde1))
    
            
    # def Q1(x):
    #     return (x - rmtilde1).T @ iCmtilde11 @ (x - rmtilde1)
    
    # def Q2(x, y):
    #     return (y - b(x)).T @ iA @ (y - b(x))
    
    # K0 = (G @ m0 - d).T @ np.linalg.inv((G @ Cm @ G.T) + Cd) @ (G @ m0 - d)
    # lbounds = np.ones(len(m0)) * bounds[0]
    # ubounds = np.ones(len(m0)) * bounds[1]
    # kb, error = scipy.stats.mvn.mvnun(lbounds, ubounds, mtilde, Cmtilde)
    # kb = 1./kb
    # Kb = kb * np.exp(-K0/2)
    
    # # print("b(m1)=", b(m1), "iA=", iA)
    # tmvn = scipy.stats.mvn.mvnun(np.array([bounds[0]]), np.array([bounds[1]]), b(m_value), iA)[0]
    
    # return Kb * np.exp(-0.5 * (np.array([(m_value - rmtilde1).T @ iCmtilde11]) @ (m_value - rmtilde1))) * tmvn 
    # return mtilde
    # Computes inverse
    iCd = np.linalg.inv(Cd)
    iCm = np.linalg.inv(Cm)
        
    # Index of the last element 
    p = np.shape(Cm)[0] - 1 
        
    # Reorganizes m0
    rm0 = np.concatenate((m0[ind_m1], np.delete(m0, ind_m1)))
    rm0 = rm0.reshape(-1, 1) # to be a column vector
        
    # Computes Cmtilde and mtilde
    Cmtilde = np.linalg.inv((G.T @ iCd @ G) + iCm)    
    mtilde = np.linalg.inv((G.T @ iCd @ G) + iCm) @ ((G.T @ iCd @ d) + iCm @ rm0)
    
    # Reorganise Cmtilde
    rCmtilde = np.copy(Cmtilde)  # initialisation
        
    if ind_m1 == 0:  # if it is the first value, no need to reorganise
        pass 
        
    elif ind_m1 == p :  # if it is the last value of m
        rCmtilde[1:ind_m1, 0] = np.copy(Cmtilde[1:ind_m1, ind_m1])
        rCmtilde[1:ind_m1, ind_m1] = np.copy(Cmtilde[1:ind_m1, 0])
            
        rCmtilde[0, 1:-1] = np.copy(Cmtilde[ind_m1, 1:-1])
        rCmtilde[ind_m1, 1:-1] = np.copy(Cmtilde[0, 1:-1])
        
    elif ind_m1 == 1 :  # if it is the second one
        rCmtilde[1+ind_m1, 0] = np.copy(Cmtilde[1+ind_m1, ind_m1])
        rCmtilde[1+ind_m1, ind_m1] = np.copy(Cmtilde[1+ind_m1, 0])
    
        rCmtilde[0, 2:] = np.copy(Cmtilde[ind_m1, 2:])
        rCmtilde[ind_m1, 2:] = np.copy(Cmtilde[0, 2:])
            
    else :
        rCmtilde[1:ind_m1, 0] = np.copy(Cmtilde[1:ind_m1, ind_m1])
        rCmtilde[1:ind_m1, ind_m1] = np.copy(Cmtilde[1:ind_m1, 0])
            
        rCmtilde[ind_m1+1, 0] = np.copy(Cmtilde[ind_m1+1, ind_m1])
        rCmtilde[ind_m1+1, ind_m1] = np.copy(Cmtilde[ind_m1+1, 0])
            
        rCmtilde[0, 1:ind_m1] = np.copy(Cmtilde[ind_m1, 1:ind_m1])
        rCmtilde[ind_m1, 1:ind_m1] = np.copy(Cmtilde[0, 1:ind_m1])
            
        rCmtilde[0, ind_m1+1:] = np.copy(Cmtilde[ind_m1, ind_m1+1:])
        rCmtilde[ind_m1, ind_m1+1:] = np.copy(Cmtilde[0, ind_m1+1:])
         
    # Switch diagonal values
    rCmtilde[0,0] = np.copy(Cmtilde[ind_m1, ind_m1])
    rCmtilde[ind_m1, ind_m1] = np.copy(Cmtilde[0,0])
        
    # Partitioned Cmtilde
    Cmtilde11 = rCmtilde[0,0]
    Cmtilde21 = rCmtilde[0, 1:]
    Cmtilde22 = rCmtilde[1:, 1:]
    Cmtilde12 = rCmtilde[1:, 0]
    Cmtilde12 = Cmtilde12.reshape(-1, 1)  # to have a column vector
        
    # Inverse of Cmtilde11
    if isinstance(Cmtilde11, (float, int)):  # if Cmtilde11 is not an array        
        iCmtilde11 = np.array([1. / Cmtilde11])
    else :
        iCmtilde11 = np.linalg.inv(Cmtilde11)
        
    # Reorganise mtilde
    rmtilde = np.copy(mtilde)
    rmtilde[0] = mtilde[ind_m1]
    try:
        rmtilde[1:] = np.delete(mtilde, ind_m1).reshape(np.shape(rmtilde[1:]))
    except ValueError:
        rmtilde[1:] = np.delete(mtilde, ind_m1)
        
    # Partitioned rmtilde
    rmtilde1 = rmtilde[0]
    rmtilde2 = rmtilde[1:] 
    
    # Computes A, iA and defines b(x)
    # if isinstance(Cmtilde12.T @ iCmtilde11, (float, int, np.float64, np.int64, np.float32, np.int32)):
    if (len(Cmtilde) == 1 and len(iCmtilde11) ==1):
    # If Cmtilde12 and Cmtilde11 are arrays of one element
        A = Cmtilde22 - (np.array([Cmtilde12.T @ iCmtilde11]) @ Cmtilde12)
        iA = 1. / A
        
        def b(x):
            return rmtilde2 + (np.array([Cmtilde12.T @ iCmtilde11]) @ (x - rmtilde1))
    if len(iCmtilde11) == 1:
        A = Cmtilde22 - ((Cmtilde12.T * iCmtilde11) @ Cmtilde12)
        iA = np.linalg.inv(A)
        def b(x):
            return rmtilde2 + ((Cmtilde12.T * iCmtilde11) * (x - rmtilde1))
    else :
    # If Cmtilde12 and Cmtilde11 are not arrays
        A = Cmtilde22 - (Cmtilde12.T @ iCmtilde11 @ Cmtilde12)
        iA = np.linalg.inv(A)
        def b(x):
            return rmtilde2 + (Cmtilde12.T @ iCmtilde11 @ (x - rmtilde1))
        
                
    def Q1(x):
        return (x - rmtilde1).T @ iCmtilde11 @ (x - rmtilde1)
        
    def Q2(x, y):
        return (y - b(x)).T @ iA @ (y - b(x))
        
    K0 = (G @ m0 - d).T @ np.linalg.inv((G @ Cm @ G.T) + Cd) @ (G @ m0 - d)
    lbounds = np.ones(np.shape(G)[1]) * bounds[0]
    ubounds = np.ones(np.shape(G)[1]) * bounds[1]
    kb, error = scipy.stats.mvn.mvnun(lbounds, ubounds, mtilde, Cmtilde)
    kb = 1./kb
    # Kb = kb * np.exp(-K0/2)
    Kb = kb
    
    lbounds = np.ones(np.shape(G)[1]-1) * bounds[0]
    ubounds = np.ones(np.shape(G)[1]-1) * bounds[1]    
    # print("b(m1)=", b(m1), "iA=", iA)
    # tmvn = scipy.stats.mvn.mvnun(np.array([bounds[0], bounds[0]]), np.array([bounds[1], bounds[1]]), b(m_value), iA)[0]
    tmvn = scipy.stats.mvn.mvnun(lbounds, ubounds, b(m_value), iA)[0]
        
    return Kb * np.exp(-0.5 * (np.array([(m_value - rmtilde1).T @ iCmtilde11]) @ (m_value - rmtilde1))) * tmvn 