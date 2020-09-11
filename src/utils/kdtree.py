from scipy.spatial import KDTree
import numpy as np
import faiss

# class KNN(object):

def KNN(lidar_data, lidar_size=(512, 448, 32), k=1, distance=75):
        points = np.where(lidar_data > 0)
        indx = np.array(list(map(lambda i: [points[0][i], points[1][i], points[2][i]], range(len(points[0])))))
        xb = indx.astype('float32')
        xq = np.array([(i, j, h) for i in range(lidar_size[0]) for j in range(lidar_size[1]) for h in range(lidar_size[2])]).astype('float32')

        index = faiss.IndexFlatL2(3)   # build the index
        index.add(xb) 

        D, I = index.search(xq, k) 
    

        result = xq[I].reshape((-1, 3))
        result = np.array(list(map(lambda p: [p[0], p[1], p[2], 1], result)))
        result = result.reshape((lidar_size[0], lidar_size[1], lidar_size[2], k, 4))

        return result