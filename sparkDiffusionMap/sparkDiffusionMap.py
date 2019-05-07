import numpy as np
import scipy
from scipy.sparse import linalg

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg.distributed import DenseMatrix
from pyspark.mllib.linalg import SparseMatrix
from math import exp
import math


class distributed_diffusion_map:
    def __init__(self):
        self.steps = -1
        self.epsilon = -1
        self.rowcount = -1
        
    @classmethod
    def create_map(cls, steps = -1, epsilon=-1):
        x = cls()
        x.steps = steps
        x.epsilon = epsilon
        return x
    
    def gaussian_kernel(self, x, y):
        #gaussian_kernel
        #This is the function that actually calculates the gaussian probablity of transition between points x and y
        #Paramaters
        #self: A copy of the diffusion map object
        #x: a 3 dimensional point 
        #y: a 3 dimensional point
        #return a real value equalling the gaussian probablity of transition between points x and y

        d=.5
        return exp(-1/self.epsilon*((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2))/d
        
    
    def rdd_row(self, point1):
        #rdd_row
        #This is the function that calculates the gaussian transition probability for a 
        #given row of the initial transition matrix then scales all the values in the the matrix to 1
        #Paramaters
        #self: A copy of the diffusion map object
        #point1: a 3 dimensional point
        #return a scaled row of the initial transition matrix

        x=[]
        for point2 in self.points:
            x.append(self.gaussian_kernel(point1,point2))
        total = math.fsum(x)
        return([y / total for y in x])
        
    def make_similarity_matrix_2(self, data_points, sc):
        #make_similarity_matrix_2
        #This function creates the initial scaled transition matrix for all combination of two points.
        #Using the BlockMatrix object allows us to potentially distribute the computation across an entire cluster of machiens
        #After the initial scaled simalarity matrix is created we then subtract it by an NxN identity matrix to weight aganist points staying at the same location
        #Finally this is stored as a sim object for the diffusion map
        #Paramaters
        #self: A copy of the diffusion map object
        #data_points: an RDD containing all the points to create the transition matrix between
        #sc: the spark context of the spark cluster
        #return: an NxN BlockMatrix containing the scaled similarity matrix for the data points passed in
        
        
        self.points = data_points
        blocks = sc.parallelize([((0,0),Matrices.dense(len(self.points), len(self.points), sc.parallelize(data_points).flatMap(lambda x : self.rdd_row(x)).collect()))])        
        
        
        mat = BlockMatrix(blocks, len(self.points), len(self.points))
        
        row_count = len(self.points)
        normed = [0]*(row_count*row_count)
        x=0
        while x < len(normed):
            normed[x] = -1
            x += 1
            x += row_count
            
        new_blocks = BlockMatrix(sc.parallelize([((0,0),Matrices.dense(row_count, row_count, normed))]), row_count, row_count)
        mat = mat.add(new_blocks)
        
        self.sim = mat
        self.row_count = self.sim.numRows()
        return self.sim
    
    def make_d_matrix(self, sc):
        #make_d_matrix
        #This is the function that creates the d matrix, which has the diagonal elements of the transition matrix. 
        #This matrix is then saved to the sparse object for the diffusion map.
        #Since this is a diagonal matrix with mostly 0 elements we use the Spare Matrix Datatype.
        #Paramaters
        #self: A copy of the diffusion map object
        #sc: The spark context
        #return the NxN sparse D Matrix

        x=0
        diag_values = []
        temp = self.sim.blocks.first()[1].values
        while x < self.row_count**2:
            diag_values.append(temp[x]**(self.steps*-1))
            x += 1
            x += self.row_count    
        self.sparse=SparseMatrix(numRows = self.row_count, numCols = self.row_count, colPtrs = range(0, self.row_count+1), rowIndices = range(0, self.row_count), values = diag_values)
        return self.sparse
    
    def normalize_matrix(self, sc):
        #normalize_matrix
        #This is the function that creates the final diffusion matrix by creating the initial MT matrix then raising it to itself for the requested number of steps
        #Paramaters
        #self: A copy of the diffusion map object
        #sc: the spark context
        #return the final NxN transition matrix

        spare_d =  self.sparse.toDense()
        block_d = BlockMatrix(sc.parallelize([((0, 0), spare_d)]), spare_d.numRows, spare_d.numRows)
        self.M1 = self.sim.multiply(block_d)
        self.MT = self.M1
        for x in range(1, self.steps):
            #This loops steps the transition matrix through the number of requested steps
            self.MT = self.MT.multiply(self.M1)
        return self.MT        
    
    def calculate_eigenvalues_v2(self):
        #This is the function that calculates the eigenvalues and eigenvectors of the transition matrix
        #Paramaters
        #self: A copy of the diffusion map object
        #return a tuple with the eigenvalues and the three eigenvectors
        
        data = self.sim.blocks.flatMap(lambda block: block[1].values).collect()
        
        chunks = [data[x:x+self.sim.numCols()] for x in range(0, len(data), self.sim.numCols())]
        
        n_evecs = 2
        
        evals, evecs = scipy.sparse.linalg.eigs(np.array(chunks), k=n_evecs+1, which='LR')

        ix = evals.argsort()[::-1][1:]

        self.eigenvalues = np.real(evals[ix])

        self.eigenvectors = np.real(evecs[:, ix])

        return self.eigenvalues, self.eigenvectors

    def get_transformed_data(self):
        #This function calculates the final diffusion map from the previously calculated eigenvalues and
        #eigenvectors then returns this matrix
        #Paramaters
        #self: A copy of the diffusion map object
        #return an  NxN matrix with the final diffusion map
        
        self.dmap = np.dot(self.eigenvectors, np.diag(self.eigenvalues))
        return self.dmap
