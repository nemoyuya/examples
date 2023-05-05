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
