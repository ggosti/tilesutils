import os
import sys
import glob
from skimage import io
import numpy as np
import cv2

def nothing(x):
    pass

ref = sys.argv[1]
print "Name of ref: ",  ref

import os
cwd = os.getcwd()

os.chdir(cwd)
folders = [name for name in os.listdir(".") if (os.path.isdir(name)) and (ref in name) and ('day' in name)]
folders.sort()
#print folders
  
tiles2folder = {} 
ltiles = []
for f in folders:
    lt = glob.glob(f+'/*tile-small-bgsub.tif')
    ltiles = ltiles + lt
    if len(lt) >= 1:
        tiles2folder[lt[0]] = f
#ltiles.sort()

for f,l in zip(folders,ltiles):
    print f,l

ims = [cv2.imread(imPath,cv2.IMREAD_UNCHANGED) for imPath in ltiles]
#print ims
shapes = np.array([ims[i].shape for i in range(len(ims))])
#print shapes
r = np.max(shapes[:,0])
c = np.max(shapes[:,1])
#print r,c
l = len(ims)
stack = np.zeros((l,r,c), dtype=ims[0].dtype)
for t,im in enumerate(ims):
    rp,cp = im.shape
    stack[t,:rp,:cp] = im

#cv2.namedWindow('Stack', cv2.WINDOW_NORMAL)
#cv2.createTrackbar('t', 'Stack', 0, l-1, nothing)

#while(0):#while(1):
#    t = cv2.getTrackbarPos('t','Stack')
#    cv2.imshow('Stack', stack[t,:,:])
#    k = cv2.waitKey(1) & 0xFF
#    if k == 27:  
#        break
#cv2.destroyAllWindows()

directory = 'small-stack-'+ref+'-bgsub/' 
if not os.path.exists(directory):
    os.makedirs(directory)
for fn,im in zip(ltiles,ims):
    #print fn,tiles2folder[fn]
    cv2.imwrite(directory + tiles2folder[fn]+'.tif',im)    


tiles2folder = {} 
ltiles = []
for f in folders:
    lt = glob.glob(f+'/*tile-bgsub.tif')
    ltiles = ltiles + lt
    #print f,lt
    if len(lt) >= 1:
        tiles2folder[lt[0]] = f

directory = 'stack-'+ref+'-bgsub/'
if not os.path.exists(directory):
    os.makedirs(directory)
for fn in ltiles:
    im = cv2.imread(fn,cv2.IMREAD_UNCHANGED)
    cv2.imwrite(directory + tiles2folder[fn]+'.tif',im) 