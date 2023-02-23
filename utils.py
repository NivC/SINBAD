
from __future__ import print_function

import numpy as np
import faiss
from sklearn.covariance import ShrunkCovariance




class kNN_shrunk:
    def __init__(self, target, K, is_cpu, is_whitening, is_vector = False, shrinkage_factor = 0.1):

        self.K = K
        self.is_whitening = is_whitening
        self.target = target
        self.shrinkage_factor = shrinkage_factor


        if is_whitening:
            cov = ShrunkCovariance(shrinkage = self.shrinkage_factor).fit(target).covariance_
            try:
                cov_inv = np.linalg.inv(cov)
            except:
                cov_inv = np.linalg.pinv(cov)
            target = target.dot(cov_inv)
            self.cov_inv = cov_inv

        res = faiss.StandardGpuResources()

        if is_vector:
            index = faiss.IndexFlatL2(target.shape[1])
        else:
            index = faiss.IndexFlatL2(target.shape[2])

        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

        if is_vector is not True:
            target =  np.concatenate(target,0)

        if is_cpu or is_vector:
            self.gpu_index = index


        self.gpu_index.add(np.ascontiguousarray(target.astype('float32')))

    def train(self, type):

        pass

    def score(self, src, is_return_ind = False):
        #print("src",src.shape)

        if self.is_whitening:
            src = src.dot(self.cov_inv)


        D, I = self.gpu_index.search(np.ascontiguousarray(src.astype('float32')), self.K)

        if is_return_ind:
            return D, I
        else:
            return D
