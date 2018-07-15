import scipy.misc as misc
from matplotlib.pyplot import *
from numpy import diag
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

class img_compression:
    def __init__(self, img):
        self.img = img
        self.k = None
        self.u, self.s, self.v = svd(img, full_matrices=True)
        self.S = np.zeros((self.u.shape[0], self.v.shape[0]))
        self.compress_img = None

    def do_compression(self, k,display):
        self.k = k
        self.S[:k, :k] = diag(self.s[:k])
        self.compress_img = np.dot(self.u, np.dot(self.S, self.v))
        if(display):
            imshow(self.compress_img)
            title("Singular values chosen:"+str(k)+" / "+str(np.count_nonzero(self.s)))
            show()

    def compression_ratio(self):
        if (self.k == None):
            print("k not initialized")
            return None
        rank = np.count_nonzero(self.s)
        return (1 - ((2 * self.k * self.img.shape[0] + self.k) / (2 *rank*self.img.shape[0]  + self.s.shape[0])))

    def dist_to_origin(self):
        return norm(self.img - self.compress_img)


def main():
    k_arr = {10:0, 30:0, 50:0, 80:0, 300:0, 512:0}
    img = misc.ascent()
    compression_obj = img_compression(img)
    flag = False
    index = [i for i in range(img.shape[0])]
    ratio_lst = list()
    dist_lst = list()
    for k in range(img.shape[0]):
        if k in k_arr:
            flag = True
        compression_obj.do_compression(k, flag)
        ratio = compression_obj.compression_ratio()
        dist = compression_obj.dist_to_origin()
        ratio_lst.append(ratio)
        dist_lst.append(dist)
        print("k:" + str(k) +
              ", compression ratio:" + str(ratio) +
              ", Frobenius distance to origin: " + str(dist))
        flag = False
    plt.figure(1)
    plt.plot(index,ratio_lst )
    plt.xlabel("Matrix Rank")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio")
    plt.show()
    plt.figure(2)
    plt.plot(index, dist_lst)
    plt.xlabel("Matrix Rank")
    plt.ylabel("% Frobenius Distance")
    plt.title("Frobenius Distance")
    plt.show()

if __name__ == "__main__":
    main()
