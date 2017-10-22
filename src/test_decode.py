'''
Created on Mar 15, 2017

@author: busta
'''

import sys
sys.path.append('/home/busta/git/yolo-text/build')

import cmp_trie
import numpy as np


if __name__ == '__main__':
    
    ts = np.random.rand(10, 150)
    
    cmp_trie.load_dict('/home/busta/data/icdar2013-Train/perImage/voc_img_1.txt')
    
    ret = cmp_trie.decode_sofmax(ts)
    print(ret)