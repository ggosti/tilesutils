#!/usr/bin/env python

import glob, os
import sys

def rename(dirFiles, pattern, newPattern):
    os.chdir(dirFiles)
    for fileName in glob.glob(pattern+'*'):
        #print fileName
        os.rename(fileName, fileName.replace(pattern, newPattern))

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

if len(sys.argv) > 1:
    dirPics = sys.argv[1]
    print "Name of dir: ",  dirPics
else:
    print "Add folder: rename.py folder/"
    quit()


os.chdir(dirPics)
for fileName in glob.glob('*.scan'):
    filePat = fileName.split('.')[0]
    print('filePat',filePat)
    cwd = os.getcwd()
    print('cwd',cwd)
    newPat = cwd.split('/')[-1]
    print('newPat',newPat)

if not filePat == newPat:
    rename(cwd, filePat, newPat)

cs = []
rs = []

for fileName2 in glob.glob(cwd+'/'+newPat+'x*.tif'):
    #print fileName2
    c = int(find_between( fileName2, cwd+'/'+newPat+'x', 'y' ))
    #print fileName2, c
    cs.append(c)
    if c==1:
        try:
            r = int(find_between( fileName2, 'x1y','.tif' ))
        except ValueError:
            r = int(find_between( fileName2, 'x1y','z' ))
        rs.append(r)
        #print r
        rs.append(r)

maxCs,maxRs = max(cs),max(rs)
print maxCs,maxRs

for fileName2 in glob.glob(cwd+'/'+newPat+'x*.tif'):
    if not 'tiles' in fileName2:
        #print fileName2
        splited = fileName2.split('.')
        c = int(find_between( fileName2, cwd+'/'+newPat+'x', 'y' ))
        try:
            r = int(find_between( fileName2, 'x'+str(c)+'y','.tif' ))
        except ValueError:
            r = int(find_between( fileName2, 'x'+str(c)+'y','z' ))
        tileNum = r + maxRs*(c-1)
        newFName = ''.join(splited[:-1]) +'tiles'+str(tileNum)+'.'+splited[-1]
        #print newFName
        os.rename(fileName2, newFName)