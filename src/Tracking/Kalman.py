import numpy as np
import math

class Kalman_Filter:
    def __init__(self,dynamic,measurable):
        '''
        Fk, the state-transition model;
        Hk, the observation model;
        Qk, the covariance of the process noise;
        Rk, the covariance of the observation noise;
        Sk, Innovation (or pre-fit residual) covariance
        Yk, Innovation or measurement residual
        '''
        self.dynamic = dynamic
        self.measurable = measurable
        self.deltat = 1.0/30
        self.F = np.zeros((dynamic,dynamic))
        self.H = np.zeros((measurable,dynamic))
        self.Q =  np.zeros((dynamic,dynamic))
        self.R = np.zeros((measurable,measurable))
        self.S = None
        self.errorCovPost = np.zeros((dynamic,dynamic))# 1.
        self.statePost = np.zeros((dynamic,1))
        self.log_m_likelihood = 0.0
        #little hack if StatePre not initialised
        self.statePre = self.statePost

    def predict(self):
        self.statePost = np.reshape(self.statePost,(self.dynamic,1))
        self.statePre = np.matmul(self.F,self.statePost)
        self.errorCovPre = np.matmul(np.matmul(self.F,self.errorCovPost),self.F.transpose()) + self.Q
        return self.statePre

    def predict_seconds(self,seconds):
        F = self.F.copy()
        F[:self.measurable,self.measurable:] = np.diag(np.array([seconds]*self.measurable,dtype = np.float64))
        cov = self.errorCovPost.copy()
        Prediction = np.matmul(F,self.statePost)
        cov = np.add(np.matmul(np.matmul(F,cov),F.transpose()),seconds*self.Q/self.deltat)
        return Prediction,math.sqrt(cov[0,0])

    def correct(self,state):
        state = np.reshape(np.array(state), (self.measurable,1))
        self.Y_pre = state - self.H @ self.statePre
        self.S = self.R + self.H @ self.errorCovPre @ self.H.transpose()
        self.K = self.errorCovPre @ self.H.transpose() @ np.linalg.inv(self.S)
        self.statePost = self.statePre + self.K @ self.Y_pre
        pt1 = (np.eye(self.dynamic)-self.K @ self.H)
        pt2 = self.K @ self.R @ self.K.transpose()
        self.errorCovPost = pt1 @ self.errorCovPre @ pt1.transpose() + pt2
        self.Y_post = state - self.H @ self.statePost

    def update_log_likelihood(self):
        pt1 = self.Y_pre.transpose() @ self.S @ self.Y_pre
        pt2 = math.log(np.linalg.det(self.S)) + self.measurable * math.log(2*math.pi)
        self.log_m_likelihood = self.log_m_likelihood - 1/2*(pt1 + pt2)

    #small hack if likelihood called before correct is called
    def initialise_S(self):
        self.S = self.R + self.H @ self.errorCovPre @ self.H.transpose()

    def get_log_likelihood(self,state):
        if self.S is None:
            self.initialise_S()
        state = np.reshape(np.array(state), (self.measurable,1))
        Y_pre = state - self.H @ self.statePre
        pt1 = Y_pre.transpose() @ self.S @ Y_pre
        return pt1
