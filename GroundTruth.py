#!/usr/bin/env python
from __future__ import print_function

import sys
import os

dirEx =  os.path.abspath('.')+'/'
sys.path.append(dirEx+'lib')

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import cv2
from skimage import io
from skimage.filters import threshold_otsu,rank, gaussian
from skimage.morphology import watershed, disk, extrema
from skimage.feature import peak_local_max
from matplotlib.widgets import Slider, RadioButtons,Button
import csv

if ( len(sys.argv) != 3):
    print("\nWrong usage: ", sys.argv[0] + " picName" + " Option\n")
    sys.exit(1)

picName = str(sys.argv[1])
picName = picName.split('.')[0]
print ("\npicName is: " , picName, "\n")
Option = int(sys.argv[2])
print ("Option is: " , Option, "\n")

#original = cv2.imread(picName + '.tif',cv2.IMREAD_UNCHANGED)
original = io.imread(picName +'.tif',as_grey=True)#,cv2.IMREAD_UNCHANGED)
if original.dtype == np.float64:
    original = np.uint8(original * (2**8-1))
print ('original',original.dtype,original.shape)
im = original

if __name__ == "__main__":
    #if im.dtype == np.float32:
    #    im = np.uint32(np.round(im,decimals=0))
    #    #print (im.dtype)
    
    #lista dei parametri utilizzati
    pars = {}  
    
    #THRESHOLDING con otsu e creazione del file bianco e nero
    start_thresh = threshold_otsu(im)
    pars['thresh.val.'] = start_thresh
    #DENOISE mediano dell'immagine con dischi di diametro dd (parametro)
    dd = 3
    pars['denoised.disk.'] = dd
    #LOCAL WELL GAUSSIANO parmetro gaussiano
    wd = 6
    pars['well.disk.'] = wd
    #LOCALMAX minima distanza fra i picchi
    peakMinD = 30
    pars['peak min distance'] = peakMinD           
       
    #crea i marcatori da capo
    if (Option == 0):

        #uso dei massimi locali come marcatori
        #bisogna fare in modo che questa sia un'opzione modificabile
        thresh = start_thresh
        denoised = rank.median(im, disk(dd))
        binary = denoised > thresh
        imTemp = im.copy()
        imTemp[np.invert(binary)] = 0
        markers = peak_local_max(imTemp, min_distance=peakMinD,indices = False, threshold_abs=start_thresh)

        markers,nm = ndi.label(markers)
        #print ('nm',nm)
        #plt.figure()
        #plt.imshow(markers)
        for lab in range(1,nm+1):
            area = np.sum( (markers == lab) )
            if  area > 1:
                #print 'area',area
                idxMarkers2 = np.argwhere(markers==lab)
                #print idxMarkers2
                mc,mr = np.floor(np.mean(idxMarkers2,axis=0))
                markers[markers == lab] = 0
                markers[int(mc),int(mr)]=lab
        #plt.figure()
        #plt.title('2')
        #plt.imshow(markers)
        idxMarkers = np.argwhere(markers>0)
        np.save(picName+'-autoPoints.npy',idxMarkers)
        np.savetxt(picName+'-autoPoints.out',idxMarkers,fmt='%d')
    
    #Si serve degli output della precedente sessione 
    elif (Option == 1):
            
        #Carica i parametri della precedente sessione se esistono e in caso li sostituisce
        try:
            with open(picName + '-auto-pars.csv', "r") as File: #, newline = '\n') as File:  
                print("Loaded parameters from previous session.\n")
                reader = csv.reader(File, delimiter = ";")
                for row in reader:
                    pars[row[0]]=row[1]

            start_thresh = int(pars['thresh.val.'])
            dd = int(pars['denoised.disk.'])
            wd = int(pars['well.disk.'])
            peakMinD  = int(pars['peak min distance'])
        except IOError:
            print("No previous session data found.\nUsing default parameters.\n")  

        #carica i marcatori della precedente sessione
        pointsName = picName +'-finalPoints.npy'

        try:
            np.load(pointsName)
        except IOError:
            print("Wrong Option choice: a file named" + picName + "-finalPoints.npy do not exist.\n Try runnng the program using Option=0 instead.")
            sys.exit(1)

        idxMarkersCol = np.load(pointsName)
        idxMarkers = idxMarkersCol[:,:2]
        markers  = np.zeros_like(im)
        nm = idxMarkers.shape[0]
        for lab in range(0,nm):
            mc,mr = idxMarkers[lab,:]
            markers[int(mc),int(mr)]=lab+1
    
    else:
        print("Wrong Option choice: Option must be 0 or 1.\n 0) create a brand new file of auto points;\n 1) load existing file of final points.")
        sys.exit(1)
    
    idxHumMarkers = []     
    thresh = start_thresh
    denoised = rank.median(im, disk(dd))
    binary = denoised > thresh

    #HISTOGRAM finestra dell'istogramma
    figH, axH = plt.subplots( figsize=(3, 3))

    axH.hist(im.ravel(), bins=256)
    axH.set_title('Histogram')
    line = axH.axvline(thresh, color='r')
    plt.subplots_adjust(bottom=0.3)

    axcolor = 'lightgoldenrodyellow'

    #RESET BUTTON
    resetax = plt.axes([0.2, 0.09, 0.4, 0.05])
    hist_reset_button = Button(resetax, 'RESET', color=axcolor, hovercolor='red')

    def reset(event):
        sTh.reset()

    hist_reset_button.on_clicked(reset)

    #SLIDER: aggiunge un widget Slider  che opera sulla zona axTh
    axTh = plt.axes([0.1, 0.18, 0.6, 0.03], facecolor=axcolor)
    sTh = Slider(axTh, 'Th', 0, int(im.max()), valinit=start_thresh)

    def update(val):
        global thresh
        global binary
        thresh = int(sTh.val)
        pars['thresh.val.'] = thresh
        denoised = rank.median(im, disk(dd))
        binary = denoised > thresh
        line.set_xdata(thresh)
        axH.draw_artist(line)
        ax[1].imshow(binary, cmap=plt.cm.gray)
        fig.canvas.draw_idle()
        figH.canvas.draw_idle()

    sTh.on_changed(update)

    # process the watershed
    well = gaussian(im,wd)
    labels = watershed(-well, markers, mask = binary)
    labels[np.invert(binary)] = 0
    
    # display results
    fig, axes = plt.subplots(ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    b = ax[0].scatter(idxMarkers[:,1],idxMarkers[:,0], c=["c"]*len(idxMarkers[:,1]))
    ax[0].set_title("Original")

    #random color
    rnc = np.random.rand(256,3)
    rnc[0,:] = [0, 0, 0]
    cmaprand =  colors.ListedColormap (rnc)

    ax[1].imshow(labels, cmap=cmaprand, interpolation='nearest')
    ax[1].set_title("Segmented")

    plt.subplots_adjust(left=0.25, bottom=0.15)

    #ADD_REMOVE_BUTTONS
    state = 0

    def add_or_remove_point(event):
        global state
        global b
        global markers
        global idxMarkers
        global idxHumMarkers

        #click x-value
        if event.inaxes==ax[0]:
            xdata_click = event.xdata
            ydata_click = event.ydata
            new_xydata_point = [int(ydata_click),int(xdata_click)]
            #print new_xydata_point 
            #print event.inaxes,xdata_click,ydata_click,state
            if state == 1:
                idxHumMarkers.append(new_xydata_point)
                humPArray=np.array(idxHumMarkers)
                ax[0].scatter(humPArray[:,1],humPArray[:,0], c=["r"]*len(humPArray[:,1]))
                #b._facecolors[-1,:] = (1,0,0,1)
                lab = 1 + markers.max()
                #print (lab)
                markers[int(ydata_click),int(xdata_click)]=lab
                #ax[1].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
                fig.canvas.draw_idle() 
            elif state == -1:
                a = np.zeros( ( len(idxMarkers[:,1]) ,3) )
                a[:,:-1] = idxMarkers
                if len(idxHumMarkers)> 0:
                    humPArray=np.array(idxHumMarkers)
                    b = np.ones( ( len(humPArray[:,1]) ,3) )
                    b[:,:-1] = humPArray
                    totInx = np.vstack((a,b))
                else:
                    totInx = a
                nearest_index = (np.abs(totInx[:,1]-xdata_click)**2+np.abs(totInx[:,0]-ydata_click)**2).argmin()
                ny,nx,ng = totInx[nearest_index]
                #print ('nearest_index',nearest_index,nx,ny,ng)
                if ng == 0.0 :
                    idxMarkers = np.delete(idxMarkers,nearest_index,axis=0)
                elif ng == 1.0 :
                    del idxHumMarkers[nearest_index-len(idxMarkers[:,1])]
                    humPArray = np.delete(humPArray,nearest_index-len(idxMarkers[:,1]),axis=0)
                ax[0].clear()
                ax[0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
                ax[0].set_title("Original")
                ax[0].scatter(idxMarkers[:,1],idxMarkers[:,0], c=["c"]*len(idxMarkers[:,1]))
                if len(idxHumMarkers)> 0:
                    ax[0].scatter(humPArray[:,1],humPArray[:,0], c=["r"]*len(humPArray[:,1]))
                markers[int(ny),int(nx)]=0
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',add_or_remove_point)

    #ADD/REMOVE BUTTON
    rax = plt.axes([0.025, 0.5, 0.15, 0.2], facecolor = axcolor)
    radio = RadioButtons(rax, ('IDLE', 'ADD', 'REMOVE'), active=0, activecolor='red')

    def addRemove(label):
        global state
        if label == 'IDLE':
            state = 0
        elif label == 'ADD':
            state = 1
        elif label == 'REMOVE':
            state = -1
        plt.draw()
    
    radio.on_clicked(addRemove)
    
    #WATERSHED BUTTON
    axwatershed = plt.axes([0.025, 0.4, 0.15, 0.1])
    bwatershed = Button(axwatershed, 'WATERSHED', color = axcolor, hovercolor = 'red')
    
    def reWatershed(event):
        global labels,binary
        labels = watershed(-well, markers, mask = binary)
        #print (np.sum(labels==0))
        labels[np.invert(binary)] = 0#labels[im<=thresh] = 0 #maschera
        ax[1].clear()
        ax[1].imshow(labels, cmap=cmaprand, interpolation='nearest')
        ax[1].set_title("Segmented")
        fig.canvas.draw_idle()      
        
    bwatershed.on_clicked(reWatershed)
      
    #crea una copia del file originale se non esiste gia' un file modificato
    temp = picName + '-modified.tif'

    try:
        io.imread(temp)
    except IOError:
        io.imsave(temp, original.copy())
    
    #LOAD BUTTON: carica il file -modified
    axloadIm = plt.axes([0.025, 0.3, 0.15, 0.1])
    bimMod = Button(axloadIm, 'LOAD' , color=axcolor, hovercolor='red')
            
    def loadIm(event):
        global im, well,binary
        im = io.imread(temp)
        denoised = rank.median(im, disk(dd))
        binary = denoised > thresh
        well = gaussian(im,wd) 
        ax[0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        b = ax[0].scatter(idxMarkers[:,1],idxMarkers[:,0], c=['c']*len(idxMarkers[:,1]))
        fig.canvas.draw_idle()   

    bimMod.on_clicked(loadIm)   

    #stampa la schermata
    plt.show()

    # PRINT POINTS
    a = np.zeros( ( len(idxMarkers[:,1]) ,3) )
    a[:,:-1] = idxMarkers
    if len(idxHumMarkers)> 0:
        humPArray=np.array(idxHumMarkers)
        b = np.ones( ( len(humPArray[:,1]) ,3) )
        b[:,:-1] = humPArray
        totInx = np.vstack((a,b))
    else:
        totInx = a
    np.save(picName + '-finalPoints.npy',totInx)
    np.savetxt(picName + '-finalPoints.out',totInx,fmt='%d')

    #SAVE PARS FILE
    with open(picName+'-auto-pars.csv', "w") as csvfile:
        w = csv.writer(csvfile, delimiter=';')
        for key, val in pars.items():
            w.writerow([key, val])
    #salva labels
    temp = picName + '-labels.tif'
    io.imsave(temp, labels.copy())