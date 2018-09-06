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
        strPat = fileName+'x'+str(x)+'y'+str(y)+'.tif'
        nList = glob.glob(cwd + strPat)
        if len(nList) == 1:
            pathIm = nList[0]
        else:
            strPat = fileName + 'x' + str(x) + 'y' + str(y) + 'tiles*.tif'
            nList = glob.glob(cwd + strPat)
            if len(nList)>1:
                print nList
                strNum = raw_input('Which (set number)?')
                pathIm = nList[strNum]
            else:
                pathIm = nList[0]
        z = 0
    else:
        strPat = fileName+'x'+str(x)+'y'+str(y)+'z*.tif'
        nList = glob.glob(cwd+strPat)
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
                z = int(find_between( pathIm, 'z','delta' ))
            except ValueError:
                z = int(find_between( pathIm, 'z','tiles' ))
    return pathIm,z

autofocus = False
#doTile = True
#show = True

parser = argparse.ArgumentParser(description='Process mosaic of z-stacks tails.')
parser.add_argument('-l', type=str,
                    help='directory with tiles')
parser.add_argument('--doTile', help='Generate mosaic with tiles', action='store_true')
parser.add_argument('--show', help='Check and modify parameters', action='store_true')
args = parser.parse_args()

print args
doTile = args.doTile
show = args.show
dirPics = args.l

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
                r = int(find_between(fileName2, 'x1y', 'tile'))
            except ValueError:
                r = int(find_between( fileName2, 'x1y','z' ))
                autofocus = True
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

#####################################################
# Check focus pparameters
#########################################################

if os.path.isfile(cwd+fileName+'-focusPoins.csv'):
    df = pd.read_csv(cwd+fileName+'-focusPoins.csv')
    print 'loaded points'
    print df
    iNames = df['img. name'].tolist()
    xs = df['x'].tolist()
    ys = df['y'].tolist()
    zs = df['z'].tolist()
    minzs = df['min z'].tolist()
else:
    xs = [1,1,maxCs]
    ys = [1,maxRs,1]
    zs = [5,5,5]
    minzs = [0,0,0]
    iNames = [None,None,None]
    for i,x,y in zip(range(1,4),xs,ys):
        ans = raw_input('Set point '+str(i)+' different from: ('+str(x)+','+str(y)+') ? (y/n/point)')
        ans = ans.split(',')
        #print ans
        if ans[0] == 'y':
            x = int(raw_input('Enter your point '+str(i)+' x: '))
            y = int(raw_input('Enter your point '+str(i)+' y: '))
            xs[i-1] = x 
            ys[i-1] = y
        elif len(ans) == 2:
            x,y = int(ans[0]),int(ans[-1])
            xs[i-1] = x 
            ys[i-1] = y
        print('Point '+str(i),x,y)
        iName,minz = getImName(autofocus,fileName,xs[i-1],ys[i-1]) #fileName+'x'+str(xs[i-1])+'y'+str(ys[i-1])+'.tif'
        iNames[i-1] =iName
        minzs[i-1] = minz
        zs[i-1] = minz + zs[i-1]
        print iName,minz
a = float(zs[2]-zs[0])/(xs[2]-xs[0]) #
b = float(zs[1]-zs[0])/(ys[1]-ys[0])
print('Points x', xs,'y',ys,'z',zs,'iNames',iNames,'min z',minzs)

##############################################################
# Check focus parameters
####################################################################

pathIm1 = iNames[0]
im1 = io.imread(pathIm1)
print
'Point1 : ', pathIm1

if show:
    chx, chy = 9, 9
    ans = raw_input('Set check point different from: (' + str(chx) + ',' + str(chy) + ') ? (y/n)')
    ans = ans.split(',')
    print
    ans
    if ans[0] == 'y':
        chx = int(raw_input('Enter your point ' + str(i) + ' x: '))
        chy = int(raw_input('Enter your point ' + str(i) + ' y: '))
    elif len(ans) == 2:
        chx, chy = int(ans[0]), int(ans[-1])
    print('Check Point ', chx, chy)

    pathIm2 = iNames[1]
    im2 = io.imread(pathIm2)
    print 'Point2 : ',pathIm2

    pathIm3 = iNames[2]
    im3 = io.imread(pathIm3)
    print 'Point3 : ',pathIm3


    pathImC, zCh = getImName(autofocus,fileName,chx,chy)
    imC = io.imread(pathImC)
    print 'Check Point : ',pathImC,zCh

    print('zstack size',im1.shape[0],im2.shape[0],im3.shape[0])


    zBase1,zBase2,zBase3= minzs

    cv2.namedWindow('Point1', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Z', 'Point1', zs[0]-zBase1, im1.shape[0]-1, nothing)
    cv2.namedWindow('Point2', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Z', 'Point2', zs[1]-zBase2, im2.shape[0]-1, nothing)
    cv2.namedWindow('Point3', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Z', 'Point3', zs[2]-zBase3, im3.shape[0]-1, nothing)
    cv2.namedWindow('Check', cv2.WINDOW_NORMAL)
    #cv2.createTrackbar('Z', 'Check', zcheck, imC.shape[0], nothing)

    # Do whatever you want with contours
    imMin1 = im1.min()
    imMax1 = im1.max()
    scale1 = 255.0/float(imMax1-imMin1)#float(mii-1)
    print(((im1-imMin1)*scale1).max() )
    imMin2 = im2.min()
    imMax2 = im2.max()
    scale2 = 255.0/float(imMax2-imMin2)#float(mii-1)
    imMin3 = im3.min()
    imMax3 = im3.max()
    scale3 = 255.0/float(imMax3-imMin3)#float(mii-1)


    imMinC = imC.min()
    imMaxC = imC.max()
    scaleC = 255.0/float(imMaxC-imMinC)
    zOld = [0,0,0,0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def showTile(wName,im,zval,zBase,imMin,scale,font=font):
        img = np.uint8((im[zval-zBase,:,:]-imMin)*scale)
        cv2.putText(img,str(zval),(10,200), font, 4,(255,255,255),2,cv2.CV_AA)
        cv2.imshow(wName, img)

    while(1):
        a = float(zs[2]-zs[0])/(xs[2]-xs[0]) #
        b = float(zs[1]-zs[0])/(ys[1]-ys[0])
        zcheck = zFun(chx,chy,xs[0],ys[0],zs[0],a,b,zCh,zCh+imC.shape[0])
        #print 'zs',zs,zcheck
        zval1 = cv2.getTrackbarPos('Z','Point1') + zBase1
        showTile('Point1',im1,zval1,zBase1,imMin1,scale1)
        zval2 = cv2.getTrackbarPos('Z','Point2') + zBase2
        showTile('Point2',im2,zval2,zBase2,imMin2,scale2)
        #cv2.imshow('Point2', np.uint8((im2[zval2-zBase2,:,:]-imMin2)*scale2) )
        zval3 = cv2.getTrackbarPos('Z','Point3') + zBase3
        showTile('Point3',im3,zval3,zBase3,imMin3,scale3)
        #cv2.imshow('Point3', np.uint8((im3[zval3-zBase3,:,:]-imMin3)*scale3) )
        showTile('Check',imC,zcheck,zCh,imMinC,scaleC)
        k = cv2.waitKey(1) & 0xFF
        zs = [zval1,zval2,zval3]
        if not (np.array(zOld) == np.array(zs + [zcheck])).all():
            print zval1,zval2,zval3
            print zcheck,zcheck-zCh
            zOld = zs + [zcheck]
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    print xs,ys,zs
    df = pd.DataFrame(
        {'img. name':iNames,
         'x': xs,
         'y': ys,
         'z': zs,
         'min z': minzs
        })

    df.to_csv(cwd+fileName+'-focusPoins.csv')

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

def stich(tile,tileS,overlap,lres,numTRows,numIRows,numTCols,numICols,angle):
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

####################################################################
# check tile parameters
#################################################################

if show:
    numTRows = 3#maxRs
    numTCols= 3#maxCs
    numIRows = im1.shape[1]
    numICols = im1.shape[2]
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
            iName,minz = getImName(autofocus,fileName,c+3,r+3)
            im = io.imread(iName)#cwd+fileName+'x'+str(c)+'y'+str(r)+'.tif')
            zVal = zFun(c,r,xs[0],ys[0],zs[0],a,b,minz,minz+im.shape[0])
            #plt.figure()
            #plt.imshow(im[int(zs),:,:])
            tileS[r-1][c-1] = im[zVal-minz,:,:]
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
            stich(tile,tileS,overlap,lres,numTRows,numIRows,numTCols,numICols,angle)
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

########################################
# make tiles
##########################################

print 'doTile',doTile
if doTile:
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

    tcols = int(numICols*numTCols*lres)
    trows = int(numIRows*numTRows*lres)  
    tile = np.zeros((trows+100,tcols+100))

    for r in range(1,numTRows+1):
        print r, numIRows*(r-1),numIRows*(r)
        for c in range(1,numTCols+1):
            #print c, numICols*(c-1),numICols*(c)
            #im = io.imread(cwd+fileName+'x'+str(c)+'y'+str(r)+'.tif')
            iName,minz = getImName(autofocus,fileName,c,r)
            im = io.imread(iName)
            zVal = zFun(c,r,xs[0],ys[0],zs[0],a,b,minz,minz+im.shape[0])
            if zVal-minz > im.shape[0]:
                print 'zVal to large!!',zVal,int(zVal),'pos',c,r,'file',iName
            if zVal-minz < 0:
                print 'zVal to small!!',zVal,int(zVal),'pos',c,r,'file',iName
            #plt.figure()
            #plt.imshow(im[int(zs),:,:])
            pasteTile(r,c,tile,im[zVal-minz,:,:] ,overlap,lres,numTRows,numIRows,numTCols,numICols,angle)

    imMinTile = tile.min()
    imMaxTile = tile.max()
    scaleTile = (2.0**16.0 - 1)/float(imMaxTile-imMinTile)
    io.imsave(cwd+fileName+'-tile.tif',np.uint16((tile-imMinTile)*scaleTile))

    from skimage.transform import rescale
    from skimage.filters import gaussian

    s = 0.1
    tile = gaussian(tile, sigma=(1.0-s)/2) 
    stile = rescale(tile, s)
    imMinTile = stile.min()
    imMaxTile = stile.max()
    scaleTile = (2.0**8.0 - 1)/float(imMaxTile-imMinTile)
    io.imsave(cwd+fileName+'-tile-small.tif',np.uint8((stile-imMinTile)*scaleTile))
    #cv2.imwrite(cwd+fileName+'-tile2.tif',tile)
    #plt.imsave(cwd+fileName+'-tile.tif',tile, cmap='gray')
