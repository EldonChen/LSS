import numpy as np
import cv2
import maxflow
import math
# pixel = np.dtype(  [('intensity',np.int8),
#                     ('r_n_link',np.float),
#                     ('d_n_link',np.float),
#                     ('s_t_link',np.float),
#                     ('t_t_link',np.float)])

"""
边界项、区域项和边的关系
E(p,q) = B(p,q)
E(p,S) = lambda * P_p('bkg') or inf or 0
E(p,T) = lambda * P_p('obj') or inf or 0

改进：B(p,q)不变
E(p,T) = C_p('obj') = max(R_p('obj'),delta*D_p(A_p))
E(p,S) = C_p('bkg') = min(R_p('obj'),delta*D_p(A_p))
"""

'''
种子点t-link权值：种子点认为是硬约束，其用户预设类别后，类别不会随分割算法而改变。
a.对于正类别种子点，s-t-link必须保留，t-t-link必须割去。工程中，通过将s-t-link权值设置为超级大值，t-t-link设置为0。保证一定仅仅割去t-t-link，否则一定不是最小割，因为当前w(s-t-link)权值是超级大值，割去这条边的代价一定是最大的。
b.反之同理。
'''
def gaussian(x,*param):
    return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
           param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))

class img_Graph:
    def __init__(self,image):
        self.shape = image.shape
        self.hight = self.shape[0]
        self.width = self.shape[1]
        self.img = image
        # self.Df = None
        # self.graph = np.empty((self.hight,self.width),dtype=pixel)
        # self.copy_intensity(image)
        self.obj_mask = None
        self.bkg_mask = None
        self.H_F = None
        self.H_B = None
        self.D_P = None
        self.R_P = None
        self.eta = 0.5
        self.delta = 1
        self.beta = -1
        self.sigma = 0
        #通过正负类种子点，我们能建立2类的颜色直方图。将直方图归一化成概率密度函数，定义为H_F，H_B。其中s-t-link权值为-ln(H_F(x))，t-t-link权值为-ln(H_B(x))，x为该像素点颜色值。


    def calculate_probability_distribution(self,mask,fit=False):
        """
        calculate the probability distribution of given mask
            args:
                mask:
            return:
                ProbDist:
            author:     Eldon_Chen, JHY_lab
            version:    1.0
        """
        ret = cv2.bitwise_and(img, img, mask=mask)
        Imax = int(img.max())
        Imin = int(img.min())
        IBin = int(Imax - Imin +1)
        Ibox = [Imin,Imax]
        hist = cv2.calcHist([ret],[0],mask,[IBin],Ibox)
        hist = np.convolve(hist.flatten(),[1/3,1/3,1/3],'same')
        hist = hist/np.sum(mask)
        H_f = lambda x : hist[x]
        if fit == True:
            x = range(IBin)
            popt,pcov = curve_fit(gaussian,x,hist,p0=[1,1,1,1,1,1])
            H_f = lambda x : gaussian(x,popt)
        return H_f

    def set_obj_mask(self,mask):
        self.obj_mask = mask
        self.H_F = self.calculate_probability_distribution(mask,fit=True)
        for i in range(self.hight):
            for j in range(self.width):
                if mask[i][j] != 0:
                    self.set_s_t_link(self.graph[i][j],np.inf)
                    self.set_t_t_link(self.graph[i][j],0)

    def set_bkg_mask(self,mask):
        self.bkg_mask = mask
        self.H_B = self.calculate_probability_distribution(mask)
        for i in range(self.hight):
            for j in range(self.width):
                if mask[i][j] != 0:
                    self.set_t_t_link(self.graph[i][j],np.inf)
                    self.set_s_t_link(self.graph[i][j],0)

    def set_Rp(self):
        self.R_P = lambda x,ap : -math.log10(self.H_F(x))/(self.H_F(x)+self.H_B(x)) if ap == 1\
                         else -math.log10(self.H_B(x))/(self.H_F(x)+self.H_B(x))

    # def copy_intensity(self, image):
    #     for i in range(self.hight):
    #         for j in range(self.width):
    #             self.graph[i][j]['intensity'] = image[i][j]

    # def set_s_t_link(self,P,value):
    #     P['s_t_link'] = value
    
    # def set_t_t_link(self,P,value):
    #     P['t_t_link'] = value

    # def set_r_n_link(self,P,value):
    #     P['r_n_link'] = value
    
    # def set_d_n_link(self,P,value):
    #     P['d_n_link'] = value

    def set_Dp(self, NearbyImg, ksize=4, k=6, m=2):
        Df = cv2.medianBlur(cv2.absdiff(self.img,NearbyImg),ksize)
        k = k * self.sigma
        D_eh = np.where(Df > k , np.power(Df / k , m) , 1)
        self.D_P = lambda x,y,ap: D_eh[x][y] if ap==1 else 1-D_eh[x][y]

    def build_graph(self):
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes(self.img.shape)
        r_n_edge = np.ones_like(self.img.shape)
        d_n_edge = np.ones_like(self.img.shape)
        s_t_edge = np.ones_like(self.img.shape)
        t_t_edge = np.ones_like(self.img.shape)
        for i in range(self.hight - 1):
            for j in range(self.width - 1):
                r_n_edge[i*self.width + j] = np.exp(selp.beta * np.power( self.img[i][j]-self.img[i][j+1] ,2))
                d_n_edge[i*self.width + j] = np.exp(selp.beta * np.power( self.img[i][j]-self.img[i+1][j] ,2))
                if self.img[i][j] == 0 or self.bkg_mask[i][j] != 0:
                    s_t_edge[i][j] = 0
                    t_t_edge[i][j] = np.inf
                elif self.obj_mask[i][j] != 0:
                    t_t_edge[i][j] = 0
                    s_t_edge[i][j] = np.inf
                else:
                    s_t_edge[i][j] = max(self.D_P(i,j,1), self.R_P(self.img[i][j],1))
                    t_t_edge[i][j] = min(self.D_P(i,j,0), self.R_P(self.img[i][j],0))

        rstructure = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [0, 0, 0]])
        dstructure = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 1, 0]])
        g.add_grid_edges(nodeids, weights=r_n_edge, structure=rstructure,symmetric=True)
        g.add_grid_edges(nodeids, weights=d_n_edge, structure=dstructure,symmetric=True)
        
        g.add_grid_tedges(nodeids, s_t_edge, t_t_edge)

    def segmentation(slef):
        

