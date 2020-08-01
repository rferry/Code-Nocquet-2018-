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
import scipy.stats


def MAP(G, d, Cd, Cm, m0, method="non-neg", bounds=(0, 10)):
    """
    Computes maximum a posteriori probability estimate. 
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


def joint_posterior(G, d, Cd, Cm, m, m0, bounds=(0, 10)):
    """
    Computes joint posterior probability of the set of parameters m.
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
    
    # COMPUTATIONS
    # Computes TMVN approximation
    Kplus, error = scipy.stats.mvn.mvnun(lb, ub, mtilde, Cmtilde)
    assert error==0, 'Potential errors in TMVN approximation'
    Kplus = 1./Kplus
    
    # Computes joint posterior
    jp = Kplus * np.exp(-0.5 * ((m - mtilde).T @ np.linalg.inv(Cmtilde) @ \
                                   (m - mtilde)))
    
    return jp


def marginal_1D(G, d, Cd, Cm, m0, m_value, ind_m1, bounds=(0, 10)):
    """
    Computes 1-D marginal probability.
    Solves Eq.11 or 12 from Njppdfocquet (2018).
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
    m0 : array, shape (p,)
        Mean of a priori Gaussian density probability function.
    m_value : float
        Value to evaluate.
    ind_m1 : int
        Index of the parameter to evaluate.
    bounds : 2-tuple or array-like, optional
        Lower and upper bounds on independent variables. Each array must have 
        shape (p,) or be a scalar, in the latter case a bound will be the same
        for all variables. The default is (0, 10).


    Raises
    ------
    ValueError
        If TLVN approximation is not working.

    Returns
    -------
    float
        Marginal probability for the value m_value.

    """
    # Converts m_value to an array
    m_value = np.array([m_value])
    
    # Inverse of Cm and Cd
    iCm = np.linalg.inv(Cm)
    iCd = np.linalg.inv(Cd)    

    # Parameters number
    p = m0.shape[0]
        
    # Reorganizes m0 to have (m0_1, m0_2)
    rm0 = np.concatenate((np.array([m0[ind_m1]]), np.delete(m0, ind_m1)))
        
    # Computes Cmtilde and mtilde
    Cmtilde = np.linalg.inv((G.T @ iCd @ G) + iCm)    
    mtilde = np.linalg.inv((G.T @ iCd @ G) + iCm) @ ((G.T @ iCd @ d) + \
                                                     iCm @ rm0)
    
    # Reorganizes Cmtilde
    rCmtilde = np.copy(Cmtilde)  # initialisation
        
    if ind_m1 == 0:  # if it is the first value, no need to reorganise
        pass 
        
    elif ind_m1 == (p-1) :  # if it is the last parameter of m
        rCmtilde[1:ind_m1, 0] = np.copy(Cmtilde[1:ind_m1, ind_m1])
        rCmtilde[1:ind_m1, ind_m1] = np.copy(Cmtilde[1:ind_m1, 0])
            
        rCmtilde[0, 1:-1] = np.copy(Cmtilde[ind_m1, 1:-1])
        rCmtilde[ind_m1, 1:-1] = np.copy(Cmtilde[0, 1:-1])
        
    elif ind_m1 == 1 :  # if it is the second one
        rCmtilde[1+ind_m1, 0] = np.copy(Cmtilde[1+ind_m1, ind_m1])
        rCmtilde[1+ind_m1, ind_m1] = np.copy(Cmtilde[1+ind_m1, 0])
    
        rCmtilde[0, 2:] = np.copy(Cmtilde[ind_m1, 2:])
        rCmtilde[ind_m1, 2:] = np.copy(Cmtilde[0, 2:])
            
    else :  # otherwise 
        rCmtilde[1:ind_m1, 0] = np.copy(Cmtilde[1:ind_m1, ind_m1])
        rCmtilde[1:ind_m1, ind_m1] = np.copy(Cmtilde[1:ind_m1, 0])
            
        rCmtilde[ind_m1+1, 0] = np.copy(Cmtilde[ind_m1+1, ind_m1])
        rCmtilde[ind_m1+1, ind_m1] = np.copy(Cmtilde[ind_m1+1, 0])
            
        rCmtilde[0, 1:ind_m1] = np.copy(Cmtilde[ind_m1, 1:ind_m1])
        rCmtilde[ind_m1, 1:ind_m1] = np.copy(Cmtilde[0, 1:ind_m1])
            
        rCmtilde[0, ind_m1+1:] = np.copy(Cmtilde[ind_m1, ind_m1+1:])
        rCmtilde[ind_m1, ind_m1+1:] = np.copy(Cmtilde[0, ind_m1+1:])
         
    # Switch diagonal values of Cmtilde 
    rCmtilde[0,0] = np.copy(Cmtilde[ind_m1, ind_m1])
    rCmtilde[ind_m1, ind_m1] = np.copy(Cmtilde[0,0])
        
    # Partitioned Cmtilde
    Cmtilde11 = rCmtilde[0,0]
    Cmtilde22 = rCmtilde[1:, 1:]
    Cmtilde12 = rCmtilde[1:, 0]
        
    # Inverse of Cmtilde11
    if isinstance(Cmtilde11, (float, int)):  # if Cmtilde11 is not an array        
        iCmtilde11 = np.array([1. / Cmtilde11])
    else :
        iCmtilde11 = np.linalg.inv(Cmtilde11)
        
    # Reorganizes mtilde
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
    if (len(Cmtilde) == 1 and len(iCmtilde11) ==1):
    # If Cmtilde12 and Cmtilde11 are arrays of one element
        A = Cmtilde22 - (np.array([Cmtilde12.T @ iCmtilde11]) @ Cmtilde12)
        iA = 1. / A
        
        def b(x):
            return rmtilde2 + (np.array([Cmtilde12.T @ iCmtilde11]) @ \
                               (x - rmtilde1))
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
        
    
    # DEALING WITH BOUNDS
    # Function to convert tuple bounds to array bounds 
    def prepare_bounds(bounds, p):
        lb, ub = [np.asarray(b, dtype=float) for b in bounds]
    
        if lb.ndim == 0:
            lb = np.resize(lb, p)
    
        if ub.ndim == 0:
            ub = np.resize(ub, p)
    
        return lb, ub    
    
    # Computes lower and upper bounds
    lb, ub = prepare_bounds(bounds, p)    
    
    # Checks that bounds are correctly definedG = np.array([[-7., -4.], [1., 10.], [2., -11.]])
d = np.array([[10.], [3.], [-5.]])
Cd = np.identity(3, dtype=float) * 5.**2
Cm = np.identity(2, dtype=float) * (0.5 / 2)**2
Cd_inv = np.linalg.inv(Cd)
Cm_inv = np.linalg.inv(Cm)
m0 = np.array([[0.5], [0.5]])
m = np.array([[0.5], [0.3]])
    if lb.shape != (p,) and ub.shape != (p,):
        raise ValueError("Bounds have wrong shape.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")
    
    # COMPUTATIONS
    # Computes TMVN approximation over the interval M+ or Mb
    kb, error = scipy.stats.mvn.mvnun(lb, ub, mtilde, Cmtilde)
    assert error==0, 'Potential errors in TMVN approximation'
    Kb = 1./kb
    
    # Lower and upper bounds over M+2 ou Mb2 interval
    lb2 = np.delete(lb, ind_m1)
    ub2 = np.delete(ub, ind_m1)
    
    # Computes TMVN approximation over the interval M+2 or Mb2
    tmvn, error = scipy.stats.mvn.mvnun(lb2, ub2, b(m_value), iA)
    assert error==0, 'Potential errors in TMVN approximation'
        
    return Kb * np.exp(-0.5 * \
                       (np.array([(m_value - rmtilde1).T @ iCmtilde11]) @ \
                        (m_value - rmtilde1))) * tmvn 


def exact_uniform(G, d, Cd, m, m0):
    """
    Computes the xact solution for the overdetermined inverse problem with 
    uniform priors.
    Solves Eq.22 from Nocquet (2018).
    
    Parameters
    ----------
    G : array, shape (n, p)
        Model matrix.
    d : array, shape (n,)
        Data vector.
    Cd : array, shape (p, p)
        Data covariance matrix.
    m : array, shape (p,)
        Parameters values to be evaluated 
    m0 : array, shape (p,)
        Mean of a priori Gaussian density probability function.


    Returns
    -------
    float
        Marginal probability for the value m_value.
    """
    
    # Inverse of Cd
    iCd = np.linalg.inv(Cd)
    
    # Determinant of Cd
    detCd = np.linalg.det(Cd)
    
    # Number of data 
    n = len(d)
    
    # Computation of Cmtilde and mtilde  
    Cmtilde = np.linalg.inv(G.T @ iCd @ G)  # Eq 22
    mtilde = Cmtilde @ G.T @ iCd @ d  # Eq 22
    
    # Computationf od Kd, K0 and Ku0
    Kd = 1. / ((2 * np.pi)**(1/n) * np.sqrt(detCd))
    K0 = d.T @ iCd @ (d - G @ mtilde)  # Eq B2
    Ku0 = np.exp(-0.5 * K0) * Kd
    

    return Ku0 * np.exp(-0.5 * ((m - mtilde).T @ np.linalg.inv(Cmtilde) @ \
                                (m - mtilde)))


def likelihood(G, d, Cd, m):
    """
    Computes the likelihood of parameters m.

    Parameters
    G : array, shape (n, p)
        Model matrix.
    d : array, shape (n,)
        Data vector.
    Cd : array, shape (p, p)
        Data covariance matrix.
    m : array, shape (p,)
        Parameters values to be evaluated 

    Returns
    -------
    float
        Likelihood of parameters m.

    """
    Cd_inv = np.linalg.inv(Cd)
    
    
    return 0.5 * (((G @ m) - d).transpose() @ Cd_inv @ ((G @ m) - d))


def multi_gauss_distrib(G, d, Cd, Cm, m, m0):
    """
    Computes multivariate Gaussian distribution.

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

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    
    # Inverse of Cm and Cd
    iCm = np.linalg.inv(Cm)  
    iCd = np.linalg.inv(Cd)
 
    # Number of data
    n= len(d)
    
    # Computation of Cmtilde and mtilde
    Cmtilde = np.linalg.inv((G.T @ iCd @ G) + iCm)
    mtilde = np.linalg.inv((G.T @ iCd @ G) + iCm) @ ((G.T @ iCd @ d) + iCm @ \
                                                     m0)
    # Inverse of Cmtilde
    iCmtilde = np.linalg.inv(Cmtilde)
    
    # Determinant of Cmtilde
    detCmtilde = np.linalg.det(Cmtilde)
    
    return (2 * np.pi)**(-n/2) * detCmtilde**(0.5)  * \
        np.exp(-0.5 * ((m - mtilde).T @ iCmtilde @ (m - mtilde)))
