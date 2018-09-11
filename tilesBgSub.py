#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from skimage import io
import os
import glob
import cv2
import sys
import pandas as pd
import subprocess

def nothing(x):
    pass

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def getImName(autofocus,fileName,x,y):
    if autofocus == False:
        pathIm = fileName+'x'+str(x)+'y'+str(y)+'.tif'
        if not os.path.isfile(pathIm):
            tileNum = r + maxRs*(c-1) 
            pathIm = fileName+'x'+str(x)+'y'+str(y)+'tiles'+str(tileNum)+'.tif'
            if not os.path.isfile(pathIm):
                print 'file name error!!!',pathIm
        z = 0
        return pathIm,z
    else:
        strPat = fileName+'x'+str(x)+'y'+str(y)+'z*.tif'
        print 'fileName',strPat
        nList = glob.glob(strPat)
        print nList
        if len(nList)>1:
            print nList
            strNum = raw_input('Which (set number)?')
            pathIm = nList[strNum]
        else:
            pathIm = nList[0]
        try:
            z = int(find_between( pathIm, 'z','.tif' ))
        except ValueError:
            try:
                z = int(find_between( pathIm, 'z','tile' ))
            except ValueError:
                z = int(find_between( pathIm, 'z','delta' ))
        return pathIm,z

autofocus = False

dirPics = sys.argv[1]
print "Name of dir: ",  dirPics

os.chdir(dirPics)
for fileName in glob.glob('*.scan'):
    fileName = '/'+fileName.split('.')[0]
    print('fileName',fileName)
    cwd = os.getcwd()
    print('cwd',cwd)

cs = []
rs = []

for fileName2 in glob.glob(cwd+fileName+'x*.tif'):
    c = int(find_between( fileName2, cwd+fileName+'x', 'y' ))
    #print fileName2, c
    cs.append(c)
    if c==1:
        try: 
            r = int(find_between( fileName2, 'x1y','.tif' ))
        except ValueError:
            try:
                r = int(find_between( fileName2, 'x1y','z' ))
                autofocus = True
            except ValueError:
                r = int(find_between( fileName2, 'x1y','tiles' ))
            #print 'autofocus',autofocus,r
        rs.append(r)
        #print r
        rs.append(r)

maxCs,maxRs = max(cs),max(rs)
print maxCs,maxRs

def zFun(x,y,x0,y0,z0,a,b,zmin,zmax):
    z = int(round(z0 + (x-x0)*a + (y-y0)*b))
    z = np.max([z,zmin])
    z = np.min([z,zmax-1])
    return z

df0 = pd.read_csv(cwd+fileName+'-focusPoins.csv')
print 'loaded points'
print df0
if 'img. name' in df0.columns: iNames = df0['img. name'].tolist()
xs = df0['x'].tolist()
ys = df0['y'].tolist()
zs = df0['z'].tolist()
if 'min z' in df0.columns:
    minzs = df0['min z'].tolist()
else:
    minzs = [0,0,0]

df = pd.read_csv(cwd+fileName+'-tilePars.csv')
print 'loaded points'
print df
angle = df['angle'].values[0]
overlap = df['overlap'].values[0]
lres = df['lres'].values[0]

print df.columns

if 'img. name' in df0.columns:
    print('Points x', xs,'y',ys,'z',zs,'iNames',iNames,'min z',minzs)
else:
    print('Points x', xs,'y',ys,'z',zs)

a = float(zs[2]-zs[0])/(xs[2]-xs[0]) #
b = float(zs[1]-zs[0])/(ys[1]-ys[0])

#try:
#    pathIm1 = cwd+fileName+'x'+str(1)+'y'+str(1)+'tiles'+str(1)+'.tif'
#    im1 = io.imread(pathIm1)
#except:
#    pathIm1 = cwd+fileName+'x'+str(1)+'y'+str(1)+'z'+str(minzs[0])+'tiles'+str(1)+'.tif'
#    im1 = io.imread(pathIm1)

pathsIm1 = glob.glob(cwd+fileName+'x'+str(1)+'y'+str(1)+'*tiles'+str(1)+'*.tif')
im1 = io.imread(pathsIm1[0])

numTRows = maxRs
numTCols= maxCs

numIRows = im1.shape[1]
numICols = im1.shape[2]
print 'Img. shape ',numIRows, numICols
#tileS = [[np.zeros( (numICols,numIRows) )] * numTRows for ncols in range(numTCols)]
indices = [[(-1,-1)] * numTCols for nrows in range(numTRows)]
tileS = [[np.zeros( (numICols,numIRows) )] * numTCols for nrows in range(numTRows)]
#print tileS
print len(tileS)

lres = 1.0
tcols = int(numICols*numTCols*lres)
trows = int(numIRows*numTRows*lres)  
tile = np.zeros((trows+100,tcols+100))

def traAndRot(im,dx,dy,angle,w,h):
    #h,w = im.shape
    Tras = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(im,Tras,(w+dx,h+dx))
    M = cv2.getRotationMatrix2D((0,0),angle,1)
    dst = cv2.warpAffine(dst,M,(w+dx,h+dx))
    return dst

def pasteTile(r,c,tile,tIm,overlap,lres,numTRows,numIRows,numTCols,numICols,angle):
    resSk = int(1.0/lres)
    w = int(numICols*(1.0-overlap)*lres)
    h = int(numIRows*(1.0-overlap)*lres)
    w2 = int(numICols*lres)
    h2 = int(numIRows*lres)         
    mask = np.ones((h+10,w+10))
    dst = traAndRot(tIm,50,50,angle,w+100,h+100)
    smallmask = traAndRot(mask,50,50,angle,w+100,h+100)
    kernel = np.ones((2,2),np.uint8)
    smallmask = cv2.erode(smallmask,kernel,iterations = 1)
    largemask = np.zeros(tile.shape)
    largemask[h*(r-1):h*r+100,w*(c-1):w*c+100] = smallmask[:h+100,:w+100]
    tile[largemask > 0] = dst[smallmask > 0]

directory = cwd+'/bestFocus/' 
if not os.path.exists(directory):
    os.makedirs(directory)

dirBgSub = cwd+'/bestFocus-bgSub/' 
if not os.path.exists(dirBgSub):
    os.makedirs(dirBgSub)

def runIJSubBack(source,target):
    com = ['java', '-jar', '/home/gosti/App/Fiji.app/jars/ij-1.52b.jar']
    com = com +['-ijpath', '/home/gosti/App/Fiji.app']
    com = com +['--headless', '--console',  '-macro', 'subBack']
    com = com +[source+'#'+target]
    DEVNULL = open(os.devnull, 'wb')
    subprocess.call(com, stdout=DEVNULL, stderr=DEVNULL)

for r in range(1,numTRows+1):
    print r, numIRows*(r-1),numIRows*(r)
    for c in range(1,numTCols+1):
        print autofocus,fileName,c,r
        iName,minz = getImName(autofocus,cwd+fileName,c,r)
        fn = iName.split('/')[-1]
        if not os.path.isfile(dirBgSub + fn):
            im = io.imread(iName)
            zVal = zFun(c,r,xs[0],ys[0],zs[0],a,b,minz,minz+im.shape[0])
            if zVal-minz > im.shape[0]:
                print 'zVal to large!!',zVal,int(zVal),'pos',c,r,'file',iName
            if zVal-minz < 0:
                print 'zVal to small!!',zVal,int(zVal),'pos',c,r,'file',iName
            #plt.figure()
            #plt.imshow(im[int(zs),:,:])
            cv2.imwrite(directory + fn, im[zVal-minz,:,:])
            #if not os.path.isfile(dirBgSub + fn):
            runIJSubBack(directory + fn,dirBgSub + fn) 
        #pasteTile(r,c,tile,im[zVal-minz,:,:] ,overlap,lres,numTRows,numIRows,numTCols,numICols,angle)

if True:
    for r in range(1,numTRows+1):
        print r, numIRows*(r-1),numIRows*(r)
        for c in range(1,numTCols+1):
            #print autofocus,fileName,c,r
            iName,minz = getImName(autofocus,dirBgSub+fileName,c,r)
            im = cv2.imread(iName,cv2.IMREAD_UNCHANGED)
            pasteTile(r,c,tile,im ,overlap,lres,numTRows,numIRows,numTCols,numICols,angle)

    imMinTile = tile.min()
    imMaxTile = tile.max()
    scaleTile = (2.0**16.0 - 1)/float(imMaxTile-imMinTile)
    io.imsave(cwd+fileName+'-tile-bgsub.tif',np.uint16((tile-imMinTile)*scaleTile))

    from skimage.transform import rescale
    from skimage.filters import gaussian

    s = 0.1
    tile = gaussian(tile, sigma=(1.0-s)/2) 
    stile = rescale(tile, s)
    imMinTile = stile.min()
    imMaxTile = stile.max()
    scaleTile = (2.0**8.0 - 1)/float(imMaxTile-imMinTile)
    io.imsave(cwd+fileName+'-tile-small-bgsub.tif',np.uint8((stile-imMinTile)*scaleTile))
    #cv2.imwrite(cwd+fileName+'-tile2.tif',tile)
    #plt.imsave(cwd+fileName+'-tile.tif',tile, cmap='gray')