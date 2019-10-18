#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import glob
import cv2
import sys
import pandas as pd
import argparse
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import median


def nothing(x):
    pass

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

#read metamorph style images from scan slide
def getImNameMM(fileName,s):
    strPat = fileName + '_w1_s'+str(s)+'*.TIF'
    nList = glob.glob(cwd + strPat)
    if len(nList)>1:
        print nList
        strNum = raw_input('Which (set number)?')
        pathIm = nList[strNum]
    else:
        pathIm = nList[0]
    return pathIm

#autofocus = False
#doTile = True
#show = True

parser = argparse.ArgumentParser(description='Process mosaic of z-stacks tails.')
parser.add_argument('-l', type=str,
                    help='directory with tiles')
parser.add_argument('-c', type=int,
                    help='num. of cols')
parser.add_argument('-r', type=int,
                    help='num. of rows')
parser.add_argument('--doTile', help='Generate mosaic with tiles', action='store_true')
parser.add_argument('--show', help='Check and modify parameters', action='store_true')
parser.add_argument('--eq', help='Equalize tile', action='store_true')
parser.add_argument('--median', help='apply median filter', action='store_true')
args = parser.parse_args()

print args
doTile = args.doTile
show = args.show
dirPics = args.l
cols = args.c
rows = args.r
equalize = args.eq
medianFilt = args.median

#dirPics = sys.argv[1]
#print "Name of dir: ",  dirPics
#if len(sys.argv)>2:
#    doTile = (sys.argv[2] == 'True')
#    if (sys.argv[2] == '-noShow'):
#        doTile = True
#        show = False
print "Do show: ",  show
print "Do Tile: ",  doTile
print dirPics

os.chdir(dirPics)
for fileName in glob.glob('*.scan'):
    fileName = '/'+fileName.split('.')[0]
    print('fileName',fileName)
    cwd = os.getcwd()
    print('cwd',cwd)

cs = []

#print cwd+fileName+'_w1_s*.TIF'
#print 'glob ',glob.glob(cwd+fileName+'_w1_s*.TIF')

for fileName2 in glob.glob(cwd+fileName+'_w1_s*.TIF'):
    pathIm1 = fileName2
    c = int(find_between( fileName2, cwd+fileName+'_w1_s', '_t' ))
    print fileName2, c
    cs.append(c)

maxCs = max(cs)
print maxCs
print 'Rows ',rows,' Cols ',cols,' tot ',rows*cols
if maxCs != rows*cols + 1:
    print 'Quite wrong cols and rows'
    quit()

###################################################
# Load tiles parameters
###################################################

if os.path.isfile(cwd+fileName+'-tilePars.csv'):
    df = pd.read_csv(cwd+fileName+'-tilePars.csv')
    print 'loaded points'
    print df
    angle = df['angle'].values[0]
    overlap = df['overlap'].values[0]
    lres = df['lres'].values[0]
else:
    angle = 0.6
    overlap = 0.10#0.08
    lres = 1.0/1.0


def traAndRot(im,dx,dy,angle,w,h):
    #h,w = im.shape
    Tras = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(im,Tras,(w+dx,h+dx))
    M = cv2.getRotationMatrix2D((0,0),angle,1)
    dst = cv2.warpAffine(dst,M,(w+dx,h+dx))
    return dst


def stitch(tile,tileS,overlap,lres,numTRows,numIRows,numTCols,numICols,angle):
    #start = 2
    #tile = np.zeros((int(numIRows*numTRows*lres),int(numICols*numTCols*lres)))
    resSk = int(1.0/lres)
    for r in range(1,numTRows+1):
        #print r, numIRows*(r-1),numIRows*(r)
        for c in range(1,numTCols+1):
            #print 'numICols',numICols,lres,float(numICols)*(1.0-overlap)*lres
            w = int(numICols*(1.0-overlap)*lres)
            h = int(numIRows*(1.0-overlap)*lres)
            w2 = int(numICols*lres)
            h2 = int(numIRows*lres)
            #print 'w',w,'h',h
            #print 'w',w*(c-1),w*(c),'h',h*(r-1),h*(r)
            #print tileS[r-1][c-1].shape
            #print 'w res',w*resSk,'h res',h*resSk,'resSk',resSk
            #print tileS[r-1][c-1][0:h*resSk:resSk,0:w*resSk:resSk].shape
            #print tile[h*(r-1):h*(r),w*(c-1):w*(c)].shape
            tIm = tileS[r-1][c-1][0:h2*resSk:resSk,0:w2*resSk:resSk]
            #print tIm.shape
            mask = np.ones((h+10,w+10))
            #mask2 = np.ones((h+1,w+1))
            dst = traAndRot(tIm,50,50,angle,w+100,h+100)
            smallmask = traAndRot(mask,50,50,angle,w+100,h+100)
            kernel = np.ones((2,2),np.uint8)
            smallmask = cv2.erode(smallmask,kernel,iterations = 1)
            #lM = M.copy()
            #lM[0,1] = lM[0,1] + w*(c-1)
            largemask = np.zeros(tile.shape)
            largemask[h*(r-1):h*r+100,w*(c-1):w*c+100] = smallmask[:h+100,:w+100]
            #print 'tile ', tile.shape#,np.sum(largemask>0),len(tile[largemask > 0])
            #print 'small ', smallmask.shape,np.sum(smallmask>0),len(dst[smallmask > 0])
            #print 'large ', largemask.shape,np.sum(largemask>0),len(tile[largemask > 0])
            #print dst.shape
            #plt.figure()
            #plt.imshow(dst)
            #plt.figure()
            #plt.imshow(smallmask)
            tile[largemask > 0] = dst[smallmask > 0]
            #plt.figure()
            #plt.imshow(tile)
            #plt.show()

# ####################################################################
# # check tile parameters
# #################################################################


im1 = io.imread(pathIm1)

print show
if show:
    numTRows = 3#maxRs
    numTCols= 3#maxCs
    numIRows = im1.shape[0]
    numICols = im1.shape[1]
    tcols = int(numICols*numTCols*lres)
    trows = int(numIRows*numTRows*lres)
    tile = np.zeros((trows+100,tcols+100),dtype=np.uint16)
    print('tile size ',tile.shape)
    print 'Img. shape ',numIRows, numICols
    #tileS = [[np.zeros( (numICols,numIRows) )] * numTRows for ncols in range(numTCols)]
    indices = [[(-1,-1)] * numTCols for nrows in range(numTRows)]
    tileS = [[np.zeros( (numICols,numIRows) )] * numTCols for nrows in range(numTRows)]
    #print tileS
    print len(tileS)

    for r in range(1,numTRows+1):
        print r, numIRows*(r-1),numIRows*(r)
        for c in range(1,numTCols+1):
            #print c, numICols*(c-1),numICols*(c)
            iName = getImNameMM(fileName,(r+1)+(c+1)*numTRows)
            im = io.imread(iName)#cwd+fileName+'x'+str(c)+'y'+str(r)+'.tif')
            #zVal = zFun(c,r,xs[0],ys[0],zs[0],a,b,minz,minz+im.shape[0])
            #plt.figure()
            #plt.imshow(im[int(zs),:,:])
            tileS[r-1][c-1] = im#[zVal-minz,:,:]
            indices[r-1][c-1] = (r,c)

    print indices
    #print tileS

    cv2.namedWindow('Tile', cv2.WINDOW_NORMAL)
    percoverlap = int(overlap*100.0)
    angleTbarVal = int(angle*10.0) + 10
    #print 'ov %',percoverlap
    cv2.createTrackbar('Overlap %', 'Tile', percoverlap , 15, nothing)
    cv2.createTrackbar('Rotate deg*10 - 10', 'Tile', angleTbarVal , 20, nothing)
    #percoverlap = cv2.getTrackbarPos('Overlap %','Tile')
    #print 'ov1 %',percoverlap

    ovOld = 0
    ovAng = -100.0
    while(1):
        percoverlap = cv2.getTrackbarPos('Overlap %','Tile')
        angleTbarVal = cv2.getTrackbarPos('Rotate deg*10 - 10','Tile')
        #print 'ov %',percoverlap
        overlap = float( percoverlap ) /100.0
        angle = float(angleTbarVal -10 ) / 10.0
        if (not ovOld == overlap) or (not ovAng == angle):
            print 'new overlap',overlap
            print 'new angle',angle
            ovOld = overlap
            ovAng = angle
            #print overlap
            tcols = int(numICols*numTCols*lres)
            trows = int(numIRows*numTRows*lres)
            tile = np.zeros((trows+100,tcols+100))
            stitch(tile,tileS,overlap,lres,numTRows,numIRows,numTCols,numICols,angle)
            if medianFilt:
                tile = median(np.uint16(tile), disk(5))
            if equalize:
                tile = exposure.equalize_hist(tile)
                imMinTile = 0
                scaleTile = 255.0
            else:
                imMinTile = tile.min()
                imMaxTile = tile.max()
                #print 'imMinTile',imMinTile,imMaxTile
                scaleTile = 255.0/float(imMaxTile-imMinTile)
        cv2.imshow('Tile', np.uint8((tile-imMinTile)*scaleTile) )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


    print overlap,angle
    df2 = pd.DataFrame(
        {'overlap': [overlap],
         'angle': [angle],
         'lres':[lres]
        })

    df2.to_csv(cwd+fileName+'-tilePars.csv')

# ########################################
# # make tiles
# ##########################################
#
# print 'doTile',doTile
# if doTile:
#     numTRows = maxRs
#     numTCols= maxCs
#     numIRows = im1.shape[1]
#     numICols = im1.shape[2]
#     print 'Img. shape ',numIRows, numICols
#     #tileS = [[np.zeros( (numICols,numIRows) )] * numTRows for ncols in range(numTCols)]
#     indices = [[(-1,-1)] * numTCols for nrows in range(numTRows)]
#     tileS = [[np.zeros( (numICols,numIRows) )] * numTCols for nrows in range(numTRows)]
#     #print tileS
#     print len(tileS)
#
#     def pasteTile(r,c,tile,tIm,overlap,lres,numTRows,numIRows,numTCols,numICols,angle):
#         resSk = int(1.0/lres)
#         w = int(numICols*(1.0-overlap)*lres)
#         h = int(numIRows*(1.0-overlap)*lres)
#         w2 = int(numICols*lres)
#         h2 = int(numIRows*lres)
#         mask = np.ones((h+10,w+10))
#         dst = traAndRot(tIm,50,50,angle,w+100,h+100)
#         smallmask = traAndRot(mask,50,50,angle,w+100,h+100)
#         kernel = np.ones((2,2),np.uint8)
#         smallmask = cv2.erode(smallmask,kernel,iterations = 1)
#         largemask = np.zeros(tile.shape)
#         largemask[h*(r-1):h*r+100,w*(c-1):w*c+100] = smallmask[:h+100,:w+100]
#         tile[largemask > 0] = dst[smallmask > 0]
#
#     tcols = int(numICols*numTCols*lres)
#     trows = int(numIRows*numTRows*lres)
#     tile = np.zeros((trows+100,tcols+100))
#
#     for r in range(1,numTRows+1):
#         print r, numIRows*(r-1),numIRows*(r)
#         for c in range(1,numTCols+1):
#             #print c, numICols*(c-1),numICols*(c)
#             #im = io.imread(cwd+fileName+'x'+str(c)+'y'+str(r)+'.tif')
#             iName,minz = getImName(autofocus,fileName,c,r)
#             im = io.imread(iName)
#             zVal = zFun(c,r,xs[0],ys[0],zs[0],a,b,minz,minz+im.shape[0])
#             if zVal-minz > im.shape[0]:
#                 print 'zVal to large!!',zVal,int(zVal),'pos',c,r,'file',iName
#             if zVal-minz < 0:
#                 print 'zVal to small!!',zVal,int(zVal),'pos',c,r,'file',iName
#             #plt.figure()
#             #plt.imshow(im[int(zs),:,:])
#             pasteTile(r,c,tile,im[zVal-minz,:,:] ,overlap,lres,numTRows,numIRows,numTCols,numICols,angle)
#
#     imMinTile = tile.min()
#     imMaxTile = tile.max()
#     scaleTile = (2.0**16.0 - 1)/float(imMaxTile-imMinTile)
#     io.imsave(cwd+fileName+'-tile.tif',np.uint16((tile-imMinTile)*scaleTile))
#
#     from skimage.transform import rescale
#     from skimage.filters import gaussian
#
#     s = 0.1
#     tile = gaussian(tile, sigma=(1.0-s)/2)
#     stile = rescale(tile, s)
#     imMinTile = stile.min()
#     imMaxTile = stile.max()
#     scaleTile = (2.0**8.0 - 1)/float(imMaxTile-imMinTile)
#     io.imsave(cwd+fileName+'-tile-small.tif',np.uint8((stile-imMinTile)*scaleTile))
#     #cv2.imwrite(cwd+fileName+'-tile2.tif',tile)
#     #plt.imsave(cwd+fileName+'-tile.tif',tile, cmap='gray')
