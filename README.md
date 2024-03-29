# Utilities for Z-stack Tiles

## rename.py

``rename.py`` is a script that renames the files obtained from the z-stack experiment in the correct way according to the directory. Assuming that the name of the folder with the z-stacks is `folder/`. I would name my folder adding to the name the date, the time, and the reference, e.g., `1022-1315-day3-ref1/`. 

```
python rename.py folder/
```

## viewTiles.py

``viewTiles.py`` allows to view tiles of zstacks, define a plane with all tiles in focus. 
Make viewTiles.py executable
```
chmod +777 viewTiles.py
```
To get help:
```
viewTiles.py -h
```
To create allignment parameters without stiching tiles, where ``folder/`` is the folder with the z-stack tiles:
```
./viewTile.py -l folder/ --show
```
To create allignment parameters and stich tiles:
```
./viewTile.py -l folder/ --show --doTile
```

To use existing allignment parameters and stich tiles:
```
./viewTile.py -l folder/ --doTile
```

Unce you defined the correcte tile parameter (thus you generated the files ``folder-tilePars.csv`` and ``folder-focusPoints.csv`` with ``./viewTile.py -l folder/ --show``) 
you can generate the mosaics with the subtracted background.
```
tilesBgSub.py ref2/0521-1748-day3-ref2/
```
To get all the mosaics working directory must be the one with the folders of the z-stacks.
```
../getTiles-bgsub.py ref2
```
To generate ground truth segments, for the first time run ``./GroundTruth.py`` with 0
```
./GroundTruth.py 0528-2137-day10-ref1-1.tif 0
```
Then run ``./GroundTruth.py`` with 1
```
./GroundTruth.py 0528-2137-day10-ref1-1.tif 1
```
