'''
These scripts require the python library CommPy!

Install:

$ git clone https://github.com/veeresht/CommPy.git
$ cd CommPy
$ python3 setup.py install
'''



import numpy as np
from commpy.channelcoding import convcode as cc

class code:
    def __init__(self,d1,d2,m):
        self.d1 = d1
        self.d2 = d2
        self.m = m # Number of delay elements in the convolutional encoder
        self.generator_matrixNSC = np.array([[self.d1, self.d2]])# G(D) corresponding to the convolutional encoder
        self.trellisNSC = cc.Trellis(np.array([self.m]), self.generator_matrixNSC)# Create trellis data structure
        self.tb_depth = 5*(self.m + 1) # Traceback depth of the decoder
        self.code_rate = self.trellisNSC.k / self.trellisNSC.n # the code rate
        ## get impulse response
        self.impulse_response = self.commpy_encode_sequence(np.concatenate([np.array([1],dtype=np.int8),np.zeros([self.m],dtype=np.int8)],axis=0)).astype(np.int8)

    def commpy_encode_sequence(self,u,terminate=False):
        if terminate:
            return cc.conv_encode(u, self.trellisNSC, code_type = 'default')
        else:
            return cc.conv_encode(u, self.trellisNSC, code_type = 'default')[:-2*self.trellisNSC.total_memory]

    def commpy_encode_batch(self,u,terminate=False):
        x0 = self.commpy_encode_sequence(u[0],terminate)
        x = np.empty(shape=[u.shape[0],len(x0)],dtype=np.int8)
        x[0] = x0
        for i in range(len(u)-1):
            x[i+1] = self.commpy_encode_sequence(u[i+1],terminate)
        return x

    def commpy_decode_sequence(self,y):
        return cc.viterbi_decode(y, self.trellisNSC, self.tb_depth,'unquantized')

    def commpy_decode_batch(self,y):
        u_hat0 = cc.viterbi_decode(y[0], self.trellisNSC, self.tb_depth,'unquantized')
        u_hat = np.empty(shape=[y.shape[0],len(u_hat0)],dtype=np.int8)
        u_hat[0] = u_hat0
        for i in range(len(y)-1):
            u_hat[i+1] = cc.viterbi_decode(y[i+1], self.trellisNSC, self.tb_depth,'unquantized')
        return u_hat
    
    def zero_pad(self,u):
        return np.reshape(np.stack([u,np.zeros_like(u)],axis=1),(-1,))
    
    def encode_sequence(self,u,terminate=False):
        if terminate:
            return np.convolve(self.zero_pad(u),self.impulse_response,mode='full')[:-1] % 2
        else:
            return np.convolve(self.zero_pad(u),self.impulse_response,mode='full')[:len(u)*2] % 2
    
    def encode_batch(self,u,terminate=False):
        x0 = self.encode_sequence(u[0],terminate)
        x = np.empty((u.shape[0],x0.shape[0]),dtype=np.int8)
        x[0] = x0
        for i in range(len(u)-1):
            x[i+1] = self.encode_sequence(u[i+1],terminate)
        return x