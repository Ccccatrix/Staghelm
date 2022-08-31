import numpy as np
import math
import os
import sys
import sympy
import itertools
from math import factorial
from thewalrus import tor
from numba import jit, njit

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

@jit(nopython=True)
def BinToDecimal(B):
    L = len(B)
    sum = 0
    for i in range(L):
        sum =sum + (2**(L-1-i))*B[i]
    return sum

@jit(nopython=True)
def BinHexOct(n,x,L):
    #十进制数n 转换到任意进制x
    res = np.zeros((L,),dtype=np.int32)
    i=0
    temp = n
    while True:
        res[L-1-i] = temp % x
        temp = temp//x
        i = i + 1
        if temp==0:
            break
    return res

@jit(nopython=True)
def _click_events(Z):
    b = np.zeros((2**Z,Z),dtype = np.int32)
    a = np.arange(0,2**Z,1)
    for i in range(0,len(a)):
        b[i:] = BinHexOct(a[i],2,Z)
    return b

@jit(nopython=True)
def _Prob_ABtoC_2(Probs_A,Probs_B):
    Probs_C = np.zeros(256,dtype=np.float64)
    for i in range(256):
        for j in range(256):
            Probs_C[i|j] += Probs_A[i]*Probs_B[j]
    return Probs_C

@jit(nopython=True)
def _Prob_ABtoC(events,Probs_A,Probs_B):
    Probs_C = np.zeros(len(Probs_A),dtype=np.float64)
    N = len(events[0])
    for k in range(0,len(events)):
        events_C = events[k]
        index = np.argwhere(events_C).astype(np.int32)[:,0]
        tol = np.sum(events_C)
        A_list = np.zeros((3 ** tol, tol),dtype=np.int32)
        B_list = np.zeros((3 ** tol, tol),dtype=np.int32)
        events_A_list = np.zeros((3 ** tol, N),dtype=np.int32)
        events_B_list = np.zeros((3 ** tol, N),dtype=np.int32)
        index_A = np.zeros((3 ** tol,),dtype=np.int32)
        index_B = np.zeros((3 ** tol,),dtype=np.int32)
        for i in range(0,3 ** tol):
            index_list = BinHexOct(i,3,tol)
            for j in range(0,tol):
                if index_list[j]==0:
                    A_list[i,j] = 1
                    B_list[i,j] = 0
                if index_list[j]==1:
                    A_list[i,j] = 0
                    B_list[i,j] = 1
                if index_list[j]==2:
                    A_list[i,j] = 1
                    B_list[i,j] = 1
            events_A_list[i][index] = A_list[i]
            events_B_list[i][index] = B_list[i]
            index_A[i] = BinToDecimal(events_A_list[i])
            index_B[i] = BinToDecimal(events_B_list[i])
        Probs_C[k] = np.sum(np.multiply(Probs_A[index_A], Probs_B[index_B]))
    return Probs_C

@jit(nopython=True)
def _MD_Threshold(events,Probs):
    D = events.shape[0]
    M = events.shape[1]
    C_1 = np.zeros((M,2),dtype=np.float64)
    C_2 = np.zeros((M,M,2,2),dtype=np.float64)
    for m in range(0,M):
        for d in range(0,D):
            j = events[d,m]
            C_1[m,j] += Probs[d]
            for n in range(0,M):
                if n!=m:
                    k = events[d,n]
                    C_2[m,n,j,k] += Probs[d]
    return  C_1, C_2

class Torontonian:

    def A_s_tor(self,A,S):
        N = len(S)
        index_0 = np.argwhere(S)[0:]
        index = np.array([index_0,index_0+N]).reshape(-1)
        A_S = A[index,:][:,index]
        return A_S

    def Prob(self,sample):
        sigma_inv_s = self.A_s_tor(self.sigma_inv,sample)
        O_s = np.eye(len(sigma_inv_s)) - sigma_inv_s
        Prob = tor(O_s)/self.sqrt_det_sigma
        return Prob.real

    def Probs(self,samples):
        return np.array([self.Prob(i) for i in samples])

    def ProbsAll(self):
        N = np.array([len(self.sigma)/2]).astype(np.int32)
        Eve = self.click_events(N[0])
        return self.Probs(Eve),Eve

    def sigma_set(self,sigma):
        self.sigma = sigma
        self.sigma_inv = np.linalg.pinv(self.sigma, rcond = 1e-15)
        self.sqrt_det_sigma = np.sqrt(np.linalg.det(self.sigma))
        return self

    def sigma_SMSS(self,rall,T):
        N = len(rall)
        S = np.zeros((2 * N, 2 * N))
        S[0: N, 0: N] = np.diag(np.cosh(rall))
        S[0: N, N: 2 * N] = np.diag(np.sinh(rall))
        S[N: 2 * N,0: N] = np.diag(np.sinh(rall))
        S[N: 2 * N,N: 2 * N] = np.diag(np.cosh(rall))

        sigma_vac = np.eye(2 * N) / 2
        sigma_in = np.dot(S, np.dot(sigma_vac, S.T.conj()))
    
        matrix_1 = np.zeros((2 * N, 2 * N), dtype=complex)
        matrix_1[0: N, 0: N] = T
        matrix_1[N: 2 * N, N: 2 * N] = T.conj()
        matrix_2 = matrix_1.T.conj()

        self.sigma = np.eye(2 * N) - 0.5 * np.dot(matrix_1, matrix_2) + np.dot(matrix_1, np.dot(sigma_in, matrix_2))
        self.sigma_inv = np.linalg.pinv(self.sigma, rcond = 1e-15)
        self.sqrt_det_sigma = np.sqrt(np.linalg.det(self.sigma))
        return self

    def sigma_Thermal(self,n_mean,N,T):
        sigma_in = np.eye(2*N) * n_mean+ 1/2 * np.eye(2*N)

        matrix_1 = np.zeros((2 * N, 2 * N), dtype=complex)
        matrix_1[0: N, 0: N] = T
        matrix_1[N: 2 * N, N: 2 * N] = T.conj()
        matrix_2 = matrix_1.T.conj()

        self.sigma = np.eye(2 * N) - 0.5 * np.dot(matrix_1, matrix_2) + np.dot(matrix_1, np.dot(sigma_in, matrix_2))
        #print(type(self.sigma))
        self.sigma_inv = np.linalg.pinv(self.sigma, rcond = 1e-15)
        self.sqrt_det_sigma = np.sqrt(np.linalg.det(self.sigma))

        return self

    def Prob_Coherent(self,Distance,sample):
        A = np.exp( - np.sum(pow(abs(Distance), 2)))
        B = np.prod(np.array([factorial(i) for i in sample]))
        C = np.prod(np.array([abs(Distance[i]) ** (2 * sample[i]) for i in range(0,len(sample))]))
        return (A / B) * C

    def traceto(self):
        return -1
    
    @classmethod
    def Prob_ABtoC(self,events,Probs_A,Probs_B):
        #return _Prob_ABtoC_2(Probs_A,Probs_B)
        return _Prob_ABtoC(events,Probs_A,Probs_B)

    @classmethod
    def click_events(self,Z):
        return _click_events(Z)

    @classmethod
    def MD_Threshold(self,evnets,Probs):
        return  _MD_Threshold(evnets,Probs)

def getNM(g2,Np):
    n = sympy.symbols('n')
    m = (Np - n)
    A = (2+n)*(2+m)*(2*n*n+2*m*m+2*m*n+3*m*m*n+3*n*n*m+m*m*n*n)
    B = (1+n)*(1+m)*(2*n+2*m+m*n)*(2*n+2*m+m*n)
    C = ((A)/(B)-g2)
    if C.evalf(subs={n:0},n=5)<=0:
        print('C(0)=',C.evalf(subs={n:0},n=5),'|','getNM() false')
        return -1, -1
    res = list(sympy.solveset(C,n,sympy.Interval(0,Np)))
    m = np.float64(res[0])
    n = np.float64(res[1])
    return m,n

def get_eta(n_mean,m_mean,p):
    A = m_mean * n_mean
    B = m_mean + n_mean
    C = 1 - 1 / p
    if A ==0:
        eta = - C / B
    else:
        eta = (- B + math.sqrt(B ** 2 - 4 * A *C)) / (2 * A)
    return eta

def get_eta_3(n_mean,m_mean,l_mean,p):
    eta = sympy.symbols('eta')
    fun_get_eta = p * (1+ n_mean * eta ) * (1+ m_mean * eta ) * (1+ l_mean * eta ) - 1
    res = list(sympy.solveset(fun_get_eta, eta, sympy.Interval(0,1)))
    return np.float64(res[0])

def get_g2(n,m):
    A = (2+n)*(2+m)*(2*n*n+2*m*m+2*m*n+3*m*m*n+3*n*n*m+m*m*n*n)
    B = (1+n)*(1+m)*(2*n+2*m+m*n)*(2*n+2*m+m*n)
    return A/B

def get_taueta(n,m,Pr):

    eta = sympy.symbols('eta')
    tau = []
    for i in range(0,len(Pr)):
        tau.append(sympy.symbols('tau_'+str(i)))
    fun = []
    for i in range(0,len(Pr)):
        fun_i = Pr[i,0] * (1 + m * tau[i] * eta) * math.exp(n * tau[i] * eta) - 1
        fun.append(fun_i)
    para = tau
    para.append(eta)

    return sympy.solve(fun, para)

@jit(nopython=True)
def get_TVD(Pr,Pr_throry):
    sum = 0
    for i in range(len(Pr)):
        sum = sum + abs(Pr[i]-Pr_throry[i])/Pr_throry[i]
    return sum

@jit(nopython=True)
def get_tau_i_3(a,b,c,p):
    #resolve (1+ax)(1+bx)(1+cx)-1/p = 0
    A = a * b * c
    B = a * b + b * c + c * a
    C = a + b + c
    D = 1 - 1 / p

    e = 1e-6

    t_0 = 0.125
    t_1 = t_0 - ((A * t_0 ** 3 + B * t_0 ** 2 + C * t_0 + D) / (3 * A * t_0 ** 2 + 2 * B * t_0 + C))
    while abs(t_1-t_0)>e:
        t_0 = t_1
        t_1 = t_0 - ((A * t_0 ** 3 + B * t_0 ** 2 + C * t_0 + D) / (3 * A * t_0 ** 2 + 2 * B * t_0 + C))
    return t_1

def get_tau_3(a,b,c,Pr_Pi_0):
    tau = np.zeros(8,dtype=np.float64)
    for i in range(8):
        tau[i] = get_tau_i_3(a,b,c,Pr_Pi_0[i])
    return tau

def get_tau_i(a,b,p):
    A = a * b
    B = a + b
    C = 1 - 1 / p
    if A == 0:
        if B!=0:
            tau_i = - C / B
        if B==0:
            tau_i = 1/8
    else:
        tau_i = (- B + math.sqrt(B ** 2 - 4 * A *C)) / (2 * A)
    return tau_i

def get_tau(a,b,Pr_Pi_0):
    N = len(Pr_Pi_0)
    tau = np.zeros(N,dtype=np.float64)
    for i in range(N):
        tau[i] = get_tau_i(a,b,Pr_Pi_0[i])
    return tau

@njit(cache=True)
def eff_corr(eff, N):
    #死时间效率修正
    L = 2 ** N
    events = _click_events(N)
    Trans = np.zeros((L, L), dtype = np.float64)
    Trans[0][0] = 1
    for i in range(1, L):
        for j in range(1, L):
            if (i|j) == i:
                index_i = np.argwhere(events[i] == 1)
                if events[j][index_i[0]] == 1:#排除掉开头不是1的子事件
                    Trans[i][j] = 1

                    for k in index_i[1:]:#计算i to j 的概率

                        if events[i][k - 1] == 1:

                            if events[j][k - 1] == 1:
                                if events[j][k] == 1:
                                    Trans[i][j] = Trans[i][j] * eff
                                if events[j][k] == 0:
                                    Trans[i][j] = Trans[i][j] * (1 - eff)
                            
                            #模型1
                            if events[j][k - 1] == 0:
                                if events[j][k] == 1:
                                    Trans[i][j] = Trans[i][j] * eff
                                if events[j][k] == 0:
                                    Trans[i][j] = Trans[i][j] * (1 - eff)

                            #模型2
                            # if events[j][k - 1] == 0:
                            #     if events[j][k] == 1:
                            #         Trans[i][j] = Trans[i][j] * 1
                            #     if events[j][k] == 0:
                            #         Trans[i][j] = 0
                            #         break

                        if events[i][k - 1] == 0:

                            if events[j][k - 1] == 1:
                                if events[j][k] == 1:
                                    Trans[i][j] = 0
                                    break
                                if events[j][k] == 0:
                                    Trans[i][j] = 0
                                    break
                                    
                            if events[j][k - 1] == 0:
                                if events[j][k] == 1:
                                    Trans[i][j] = Trans[i][j] * 1
                                if events[j][k] == 0:
                                    Trans[i][j] = 0
                                    break         

    return Trans.T

@njit(cache=True)
def BinReverse(num, mode):
    event = BinHexOct(num,2,mode)
    res = BinToDecimal(event[::-1])
    return res

@njit(cache=True)
def BinReverseIndex(mode):
    L = 2 ** mode
    Rev_index = np.zeros((L, 1),dtype = np.int32)
    for i in range(0, L):
        Rev_index[i][0] = BinReverse(i, mode)
    return Rev_index
