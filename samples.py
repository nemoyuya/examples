# -*- coding: utf-8 -*-

def imread_ja(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    """
    日本語が混ざったパスのcv2.imread
    """
    
    import numpy as np
    import cv2
    import os
    
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite_ja(filename, img, params=None):
    """
    日本語が混ざったパスのcv2.write
    """
    
    import numpy as np
    import cv2
    import os

    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def listdir(dir_path):
    """
    os.listdirの拡張版
    ファイル名のリストとファイルパスのリストを返す
    """
    
    import os

    file_names = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path,file_name) in file_names]

    return file_names, file_paths



class customTSNE(object):
    """
    openTSNEをラップしたもの
    
    openTSNE
    インストール： pip install opentsne
    ライセンス: OSI Approved (BSD-3-Clause)
    作者：Pavlin Poličar (pavlin.g.p@gmail.com)
    """
    from openTSNE import TSNE
    import numpy as np
    
    def __init__(self):
        self.train_emb = None
        self.test_emb = None

    def fit(self, data: np.array):
        """
        openTSNEのfit
        """
        self.train_emb = TSNE().fit(data)

    def transform(self, data: np.array):
        """
        openTSNEのtransform
        """
        self.test_emb = self.train_emb.transform(data)

    def plot_train(self, label = None):
        """
        matplotlibでのplot
        """        
        plt.clf()
        plt.scatter(self.train_emb[:,0], self.train_emb[:,1], c=label, cmap="winter")
        plt.show()

    def plot_test(self, label = None):
        """
        matplotlibでのplot
        """        
        plt.clf()
        plt.scatter(self.test_emb[:,0], self.test_emb[:,1], c=label, cmap="autumn")
        plt.show()

    def plot_both(self, train_label = None, test_label = None):
        """
        matplotlibでのplot
        """        
        plt.clf()
        plt.scatter(self.train_emb[:,0], self.train_emb[:,1], c=train_label, cmap="winter")
        plt.scatter(self.test_emb[:,0], self.test_emb[:,1], c=test_label, cmap="autumn")
        plt.show()



class customUMAP(object):
    """
    umap-learnをラップしたもの
    
    umap-learn
    インストール： pip install opentsne
    ライセンス: OSI Approved (BSD-3-Clause)
    作者：Pavlin Poličar (pavlin.g.p@gmail.com)
    """
    from umap import UMAP
    import numpy as np
    
    def __init__(self):
        self.train_emb = None
        self.test_emb = None
        self.mapper = UMAP()

    def fit(self, data: np.array):
        """
        openTSNEのfit
        """
        self.mapper.fit(data)
        self.train_emb = self.mapper.transform(data)

    def transform(self, data: np.array):
        """
        openTSNEのtransform
        """
        self.test_emb = self.mapper.transform(data)

    def plot_train(self, label = None):
        """
        matplotlibでのplot
        """        
        plt.clf()
        plt.scatter(self.train_emb[:,0], self.train_emb[:,1], c=label, cmap="winter")
        plt.show()

    def plot_test(self, label = None):
        """
        matplotlibでのplot
        """        
        plt.clf()
        plt.scatter(self.test_emb[:,0], self.test_emb[:,1], c=label, cmap="autumn")
        plt.show()

    def plot_both(self, train_label = None, test_label = None):
        """
        matplotlibでのplot
        """        
        plt.clf()
        plt.scatter(self.train_emb[:,0], self.train_emb[:,1], c=train_label, cmap="winter")
        plt.scatter(self.test_emb[:,0], self.test_emb[:,1], c=test_label, cmap="autumn")
        plt.show()
