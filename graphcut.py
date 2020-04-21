import maxflow
import numpy as np

class GraphCut:
    def __init__(self,img=None):

    def LoadImg(self,image):
        self.shape = image.shape
        self.hight = self.shape[0]
        self.width = self.shape[1]
        self.img = image
        self.graph = np.empty((self.hight,self.width),dtype=pixel)
        self.copy_intensity(image)
        self.obj_mask = None
        self.bkg_mask = None
        self.H_F = None
        self.H_B = None