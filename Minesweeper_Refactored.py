# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:01:13 2022

@author: Guilherme
"""

import random
import sys
import time
from PIL import ImageGrab
from PIL import Image
from ahk import AHK
from itertools import permutations
import numpy as np
import os
import pymem


def runMinesweeper():
    ## This function runs a new instance of the Minesweeper game if no instance is running
    ahk = AHK()
    win = ahk.find_window(title=b'Minesweeper Arbiter') # Find the opened window

    if isinstance(win, type(None)):
        os.startfile('C:\\Users\\Guilherme\\Documents\\Python Scripts\\arbiter\\ms_arbiter.exe')
        time.sleep(0.6)
        win = ahk.find_window(title=b'Minesweeper Arbiter') # Find the opened window
    
    return ahk, win

def getWindowCoordinates(win):
    ## This function returns the coordinates of the current Minesweeper window
    height = 0
    width = 0
    while ((height == 0) or (width == 0)):
        win.activate() 
        x,y,width,height = win.rect
    
    return x, y, width, height

def getGameScreenshot(win,GameWindowEdges):
    ## This function takes a screenshot of the game window and returns it as a Numpy array
    
    win.activate()
    screen_capture = ImageGrab.grab(bbox =(GameWindowEdges[0], GameWindowEdges[1], GameWindowEdges[0]+GameWindowEdges[2], GameWindowEdges[1]+GameWindowEdges[3]))
    screen_capture_array = np.array(screen_capture)
    
    return screen_capture_array

def getResetButtonPosition(screen_capture_array):
    ## This function receives as input the window screenshot as an array, and locates the smily face using the pixel information
    
    yellow_pixels = np.where((screen_capture_array[:,:,0] == 255) & (screen_capture_array[:,:,1] == 255) & (screen_capture_array[:,:,2] == 0))
    reset_button_x = yellow_pixels[1][yellow_pixels[1] > 100][0]
    reset_button_y = yellow_pixels[0][yellow_pixels[0] > 100][0]
    
    return reset_button_x, reset_button_y

def clickResetButton(ahk,win, GameWindowEdges,reset_button_x, reset_button_y):
    ## This function simply clicks on the reset button given its coordinates
    
    pixel_x = GameWindowEdges[0]+reset_button_x
    pixel_y = GameWindowEdges[1]+reset_button_y

    win.activate() 
    ahk.click(pixel_x,pixel_y) # Reset the Game 

def resetGame(win, ahk, GameWindowEdges):
    # Get the location of and click the reset button
    
    screen_capture_array = getGameScreenshot(win, GameWindowEdges)
    reset_button_x, reset_button_y = getResetButtonPosition(screen_capture_array)
    clickResetButton(ahk,win, GameWindowEdges,reset_button_x, reset_button_y)

def LocateEdge(screen_array, edge_array):
    ## This function will try to locate the edges of the grid in the screen, given the pixels array
    
    minMatrix = 9999*np.ones([6,6],dtype=int)
    for i in range(np.size(screen_array,0)-6):
        for j in range(np.size(screen_array,1)-6):
            diffMatrix = (screen_array[i:i+6,j:j+6,0] - edge_array)**2
            if np.sum(diffMatrix) < np.sum(minMatrix):
                minMatrix = diffMatrix.copy()
                i_min = i
                j_min = j
    return i_min,j_min

def CropScreenArray(screen_array, gridEdges):
    ## This function removes unnecessary pixels from the game array given the edges as inputs
    
    cropped_array = screen_array[gridEdges[1]+5:gridEdges[3]+1,gridEdges[0]+5:gridEdges[2]+1,:]
    cropped_array[cropped_array > 220] = 255 # Just in the case that the pixel values are not exactly as expected
    cropped_array[cropped_array < 40] = 0
    cropped_array[np.nonzero((cropped_array > 115) & (cropped_array < 140))] = 128
    
    return cropped_array

def getGridEdges(win, GameWindowEdges, screen_capture_array):
    # This function locates and extract the location of the four pixels lying on the edges of the grid
    
    top_edge, left_edge = LocateEdge(screen_capture_array, top_left_edge_pix)
    bottom_edge, right_edge = LocateEdge(screen_capture_array, bottom_right_edge_pix)
    gridEdges = np.array([left_edge, top_edge, right_edge, bottom_edge])
    
    return gridEdges

def convert_to_1_byte(n: int) -> int:
    ## Convert the 4-byte unsigned integer to a 1-byte integer
    
    return n & 0xff

def getInitialNumberofBombs():
    ## Reads the memory and returns the initial value of bombs
    
    pm = pymem.Pymem('ms_arbiter.exe')
    Alloc_Address = pm.base_address
    mem_value_4b = pm.read_uint(Alloc_Address+0x145C6C)
    value = convert_to_1_byte(mem_value_4b)
    
    return value

def colorToRGB(color):
    if color == "Red":
        return [255,0,0]
    elif color == "Green":
        return [0,128,0]
    elif color == "Blue":
        return [0,0,255]
    elif color == "Dark Blue":
        return [0,0,128]
    elif color == "Black":
        return [0,0,0]
    elif color == "White":
        return [255,255,255]
    elif color == "Dark Red":
        return [128,0,0]
    elif color == "Turquoise":
        return [0,128,128]
    elif color == "Gray":
        return [128,128,128]
    
def extractGameMatrix(nrows, ncols, cell_len, grid_array): 
    GameMatrix = np.zeros([nrows, ncols],dtype=int)
    
    for i in range(nrows):
        for j in range(ncols):
            if grid_array[i*cell_len,j*cell_len,0] > 220:
                # This condition represents unclicked or flags
                if (grid_array[i*cell_len + cell_len//2 + 4, j*cell_len + cell_len//2,:] == colorToRGB("Black")).all() : 
                    GameMatrix[i,j] = -88  # Flag
                else:
                    GameMatrix[i,j] = 0 # Unclicked
            else :
                if (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Black")).all():
                    GameMatrix[i,j] = -99 # Bomb
                elif (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Blue")).all():
                    GameMatrix[i,j] = 1
                elif (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Green")).all():
                    GameMatrix[i,j] = 2
                elif (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Red")).all():
                    GameMatrix[i,j] = 3
                elif (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Dark Blue")).all():
                    GameMatrix[i,j] = 4
                elif (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Dark Red")).all():
                    GameMatrix[i,j] = 5
                elif (grid_array[i*cell_len + cell_len//2, j*cell_len + cell_len//2,:] == colorToRGB("Turquoise")).all():
                    GameMatrix[i,j] = 6
                else:
                    GameMatrix[i,j] = -1
    return GameMatrix

def clickCell(chosenCell, ncols, cell_len, GameWindowEdges, gridEdges, button):
    # This function will find the location and click on the pixel corresponding to the given cell
    rownum = chosenCell//ncols
    colnum = chosenCell % ncols
    rowcenterpixel = rownum*cell_len + cell_len//2
    colcenterpixel = colnum*cell_len + cell_len//2
    pixel_x = GameWindowEdges[0]+gridEdges[0]+colcenterpixel
    pixel_y = GameWindowEdges[1]+gridEdges[1]+rowcenterpixel
    # Click
    win.activate()
    if button == 'left':
        ahk.click(pixel_x,pixel_y)
    elif button == 'right':
        ahk.right_click(pixel_x,pixel_y)

def clickInitialCell(ncols,cell_len, GameWindowEdges):
    ## Select Random Square to First Click
    chosenCell = random.randint(0,nrows*ncols-1)
    clickCell(chosenCell, ncols,cell_len,GameWindowEdges,gridEdges,'left')
    
def clickRandomCell(GameMatrix, ncols):
    # Check what cells are unclicked
    possible_cells = np.where(GameMatrix==0)
    cellnumberlist = []
    for i in range(len(possible_cells[0])):
        cellnumberlist.append(ncols*possible_cells[0][i]+ possible_cells[1][i])

    chosenCell = cellnumberlist[random.randint(0,len(cellnumberlist)-1)]
    clickCell(chosenCell, ncols,cell_len,GameWindowEdges,gridEdges,'left')

# Update the Game Matrix
def getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len):
    screen_capture_array = getGameScreenshot(win, GameWindowEdges)     
    grid_array = CropScreenArray(screen_capture_array, gridEdges)
    GameMatrix = extractGameMatrix(nrows, ncols, cell_len, grid_array)
    return GameMatrix

def listAllNeighborCells(row,col, nrows, ncols):
    ## This function returns a list of all the neighbor cells of a given cell,
    # independent of its value
    L = []
    for i in range(row-1,row+2):
        for j in range(col-1,col+2):
            if i >= 0 and j >=0 and i < nrows and j < ncols:
                if i !=row or j != col:
                    L.append([i,j])
    return L

def listPositiveElements(GameMatrix):
    # List the position of all elements in the Grid that are bigger than zero (numeric cells)
    natElem = np.where(GameMatrix > 0) 
    
    return natElem

def scanNeighborhood(GameMatrix, natElem):
    # This functions looks for all the unclicked neighbors and number of flags for each of the numeric cells
    unclickedNeighbors = [] # Initialize an empty list
    neighborFlags = np.zeros([len(natElem[0])],dtype = int)
    nrows = np.size(GameMatrix,0)
    ncols = np.size(GameMatrix,1)
    for i in range(len(natElem[0])):
        Neighbors = listAllNeighborCells(natElem[0][i], natElem[1][i], nrows, ncols)
        unclickedNeighbors.append([])
        for j in range(len(Neighbors)):
            if GameMatrix[Neighbors[j][0],Neighbors[j][1]] == 0:
                unclickedNeighbors[i].append(Neighbors[j])
            elif GameMatrix[Neighbors[j][0],Neighbors[j][1]] == -88:
                neighborFlags[i] += 1
    
    return unclickedNeighbors, neighborFlags

def findFlags(inputMatrix, natElem, unclickedNeighbors, neighborFlags):
    # Check if there is an obvious flag to be inserted and mark it into the matrix
    outputMatrix = inputMatrix.copy()
    for i in range(len(natElem[0])):
        if inputMatrix[natElem[0][i],natElem[1][i]] == neighborFlags[i]:
            for j in range(len(unclickedNeighbors[i])):
                outputMatrix[unclickedNeighbors[i][j][0], unclickedNeighbors[i][j][1]] = -22 # New clickable cell discovered
        elif inputMatrix[natElem[0][i],natElem[1][i]] == len(unclickedNeighbors[i]) + neighborFlags[i]:
            for j in range(len(unclickedNeighbors[i])):
                outputMatrix[unclickedNeighbors[i][j][0], unclickedNeighbors[i][j][1]] = -77 #New Flag discovered
    
    return outputMatrix

def lookForBombs(GameMatrix):
    natElem = listPositiveElements(GameMatrix)
    unclickedNeighbors, neighborFlags = scanNeighborhood(GameMatrix, natElem)
    updatedMatrix = findFlags(GameMatrix, natElem, unclickedNeighbors, neighborFlags)
    
    return updatedMatrix

def solveCornerMethod(win,GameWindowEdges, gridEdges, nrows, ncols, cell_len):
    ## This function checks if there are cells uncovered by the corner method, if there are, it should click or flag the respective cells
    GameMatrix = getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len)
    updatedGameMatrix = lookForBombs(GameMatrix)
    
    while np.array_equal(updatedGameMatrix,GameMatrix) is False: 
        # That means there are cells to be marked as flags
        if np.count_nonzero(updatedGameMatrix == -77) > 0: # At least one new flag was detected
                for i in range(np.size(GameMatrix,0)):
                    for j in range(np.size(GameMatrix,1)):
                        if updatedGameMatrix[i,j] == -77:
                            chosenCell = np.size(GameMatrix,1)*i + j
                            clickCell(chosenCell, ncols, cell_len, GameWindowEdges, gridEdges, 'right')
        elif np.count_nonzero(updatedGameMatrix == -22) > 0: # At least one clickable cell was detected
                for i in range(np.size(GameMatrix,0)):
                    for j in range(np.size(GameMatrix,1)):
                        if updatedGameMatrix[i,j] == -22:
                            chosenCell = np.size(GameMatrix,1)*i + j
                            clickCell(chosenCell, ncols, cell_len, GameWindowEdges, gridEdges, 'left')
        # Refresh the GameMatrix
        GameMatrix = getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len)
        updatedGameMatrix = lookForBombs(GameMatrix)
        
    return GameMatrix

def processUnclickedNeighbors(unclickedNeighbors):
# Get the unique values in the unclickedNeighbors list
    uniqueCells = []
    NonEmptyRows = []
    for i in range(len(unclickedNeighbors)):
        for j in range(len(unclickedNeighbors[i])):
            if unclickedNeighbors[i][j] not in uniqueCells:
                uniqueCells.append(unclickedNeighbors[i][j])
        if len(unclickedNeighbors[i]) > 0:
            NonEmptyRows.append(i)
    
    return uniqueCells, NonEmptyRows

def defineLinearSystem(GameMatrix, NonEmptyRows, uniqueCells, natElem, unclickedNeighbors, neighborFlags):
    # Complete the linear system equations
    controlMatrix = np.zeros([len(NonEmptyRows), len(uniqueCells)], dtype=int)
    bArray = np.zeros([len(NonEmptyRows),1],dtype = int)
    for i in range(len(NonEmptyRows)):
        for j in range(len(uniqueCells)):
                if uniqueCells[j] in unclickedNeighbors[NonEmptyRows[i]]:
                    controlMatrix[i][j] = 1
        
        bArray[i] = GameMatrix[natElem[0][NonEmptyRows[i]], natElem[1][NonEmptyRows[i]]] - neighborFlags[NonEmptyRows[i]]
        
    bArray = bArray.T.flatten() # Fix the shape of the bArray
    
    return controlMatrix, bArray

def listPossibleSolutions(missingFlags, uniqueCells):
## List all possible solutions
    if missingFlags+1 <= len(uniqueCells):
        numberofsolutions = 0
        for i in range(missingFlags+1):
            numberofsolutions += np.math.factorial(len(uniqueCells))//(np.math.factorial(len(uniqueCells)-i)*np.math.factorial(i))
        if numberofsolutions < 2000:
            listofSolutions = []
            for i in range(missingFlags+1):
                solShape = np.zeros(len(uniqueCells), dtype = int)
                solShape[0:i] = 1
                listofSolutions.append(list(set(permutations(solShape))))
            
            listofSolutions = sum(listofSolutions, []) # Flatten List to 1-D
        else:
            listofSolutions = []
    else:
        listofSolutions = []
        
    return listofSolutions

def filterValidSolutions(listofSolutions, controlMatrix, bArray):
    ## Filter only valid solutions
    validSols = []
    for i in range(len(listofSolutions)):
        if np.array_equal(bArray, np.dot(controlMatrix,listofSolutions[i])):
            validSols.append(list(listofSolutions)[i])
            
    return validSols

def findFlagsLAmethod(inputMatrix, validSols, uniqueCells):
    outputMatrix = inputMatrix.copy()           
    # Check if there is unique solutions (same value for all the possible solutions)
    if len(validSols) > 0:
        SumofSolutions = np.sum(validSols,axis=0)
        for i in range(len(SumofSolutions)):
            if SumofSolutions[i] == len(validSols): # That means this cell is a bomb
                outputMatrix[uniqueCells[i][0], uniqueCells[i][1]] = -77
            elif SumofSolutions[i] == 0: # That means this is a safe cell
                outputMatrix[uniqueCells[i][0], uniqueCells[i][1]] = -22
            
    return outputMatrix

def lookForBombsLAMethod(GameMatrix, maxbombs):
    ## In the case the corner method does not work, use the linear algebra method
    numberofFlags = np.count_nonzero(GameMatrix == -88)
    missingFlags = maxbombs - numberofFlags
    natElem = listPositiveElements(GameMatrix)
    unclickedNeighbors, neighborFlags = scanNeighborhood(GameMatrix, natElem)
    uniqueCells, NonEmptyRows = processUnclickedNeighbors(unclickedNeighbors)
    controlMatrix, bArray = defineLinearSystem(GameMatrix, NonEmptyRows, uniqueCells, natElem, unclickedNeighbors, neighborFlags)
    if len(controlMatrix) > 0:
        updatedGameMatrix = findFlagsNewMethod(GameMatrix, uniqueCells, controlMatrix, bArray)
        if np.array_equal(updatedGameMatrix,GameMatrix) is True:
            listofSolutions = listPossibleSolutions(missingFlags, uniqueCells)
            validSols = filterValidSolutions(listofSolutions, controlMatrix, bArray)
            updatedGameMatrix = findFlagsLAmethod(GameMatrix, validSols, uniqueCells)
    else:
        updatedGameMatrix = GameMatrix.copy()
    return updatedGameMatrix

def solveLAMethod(win,GameWindowEdges, gridEdges, nrows, ncols, cell_len, maxbombs):
    GameMatrix = getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len)
    updatedGameMatrix = lookForBombsLAMethod(GameMatrix, maxbombs)
    
    while np.array_equal(updatedGameMatrix,GameMatrix) is False: 
        # That means there are cells to be marked as flags
        if np.count_nonzero(updatedGameMatrix == -77) > 0: # At least one new flag was detected
                for i in range(np.size(GameMatrix,0)):
                    for j in range(np.size(GameMatrix,1)):
                        if updatedGameMatrix[i,j] == -77:
                            chosenCell = np.size(GameMatrix,1)*i + j
                            clickCell(chosenCell, ncols, cell_len, GameWindowEdges, gridEdges, 'right')
        elif np.count_nonzero(updatedGameMatrix == -22) > 0: # At least one clickable cell was detected
                for i in range(np.size(GameMatrix,0)):
                    for j in range(np.size(GameMatrix,1)):
                        if updatedGameMatrix[i,j] == -22:
                            chosenCell = np.size(GameMatrix,1)*i + j
                            clickCell(chosenCell, ncols, cell_len, GameWindowEdges, gridEdges, 'left')
        # Refresh the GameMatrix
        GameMatrix = solveCornerMethod(win,GameWindowEdges, gridEdges, nrows, ncols, cell_len)
        updatedGameMatrix = lookForBombsLAMethod(GameMatrix,maxbombs)  
    
    return GameMatrix

def isSolved(GameMatrix):
    if -99 in GameMatrix:
        return 0 # It failed
    elif 0 not in GameMatrix and -99 not in GameMatrix:
        return 1 # It succeed
    else:
        return 2 # Still not finished
    
def solvePuzzle(win, nrows, ncols, cell_len, GameWindowEdges, gridEdges, maxbombs):
    ## Try to solve the problem
    resetGame(win, ahk, GameWindowEdges)
    clickInitialCell(ncols, cell_len, GameWindowEdges)
    
    GameMatrix = getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len)
    
    while isSolved(GameMatrix) == 2: 
        GameMatrix = solveCornerMethod(win,GameWindowEdges, gridEdges, nrows, ncols, cell_len)
        GameMatrix = solveLAMethod(win,GameWindowEdges, gridEdges, nrows, ncols, cell_len, maxbombs)
        if isSolved(GameMatrix) == 2:
            GameMatrix = getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len)
            clickRandomCell(GameMatrix, ncols)
        
        GameMatrix = getGameMatrix(win, GameWindowEdges, gridEdges, nrows, ncols, cell_len)
    
    if isSolved(GameMatrix) == 0:
        output = 0
    elif isSolved(GameMatrix) == 1:
        output = 1
    
    return output

def findFlagsNewMethod(GameMatrix, uniqueCells, controlMatrix, bArray):
    # This function will try to discover the cell values by substracting the linear system rows
    M = np.zeros([np.size(controlMatrix,0)-1, np.size(controlMatrix,1),np.size(controlMatrix,0)],dtype=int)
    n = np.zeros([np.size(controlMatrix,0)-1,np.size(controlMatrix,0)], dtype = int) # Independent term per row per table
    nzcount = np.zeros([np.size(controlMatrix,0)-1,np.size(controlMatrix,0)], dtype = int) # Non-zeros per row per table
    
    for i in range(np.size(controlMatrix,0)):
        L = list(range(np.size(controlMatrix,0)))
        L.remove(i)
        for j in range(np.size(M,0)):  
            M[j,:,i] = controlMatrix[L[j],:] - controlMatrix[i,:]
            n[j,i] = bArray[L[j]] - bArray[i]
            nzcount[j,i] = np.count_nonzero(M[j,:,i])
    
    updatedGameMatrix = GameMatrix.copy()
    
    single_values_index=np.where(nzcount == 1)
    for i in range(len(single_values_index[0])//2):
        col = np.where(M[single_values_index[0][2*i], :, single_values_index[1][2*i]] != 0)
        nb = n[single_values_index[0][2*i], single_values_index[1][2*i]]//int(M[single_values_index[0][2*i],col,single_values_index[1][2*i]])
        if nb == 0:
            updatedGameMatrix[uniqueCells[col[0][0]][0], uniqueCells[col[0][0]][1]] = -22
        elif nb == 1:
            updatedGameMatrix[uniqueCells[col[0][0]][0], uniqueCells[col[0][0]][1]] = -77
    
    double_values_index=np.where(nzcount == 2)
    for i in range(len(double_values_index[0])//2):
        col = np.where(M[double_values_index[0][2*i], :, double_values_index[1][2*i]] != 0)
        nb = n[double_values_index[0][2*i], double_values_index[1][2*i]] # independent term, number of bombs
        if nb*M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][0] == -1 and nb*M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][1] == 1:
            updatedGameMatrix[uniqueCells[col[0][0]][0], uniqueCells[col[0][0]][1]] = -22
            updatedGameMatrix[uniqueCells[col[0][1]][0], uniqueCells[col[0][1]][1]] = -77
        elif nb*M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][0] == 1 and nb*M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][1] == -1:
            updatedGameMatrix[uniqueCells[col[0][0]][0], uniqueCells[col[0][0]][1]] = -77
            updatedGameMatrix[uniqueCells[col[0][1]][0], uniqueCells[col[0][1]][1]] = -22
        elif (M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][0] == 1 
              and M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][1] == 1 
              and nb == 2) or (M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][0] == -1 
                    and M[double_values_index[0][2*i],col,double_values_index[1][2*i]][0][1] == -1 
                    and nb == -2):
            updatedGameMatrix[uniqueCells[col[0][0]][0], uniqueCells[col[0][0]][1]] = -77
            updatedGameMatrix[uniqueCells[col[0][1]][0], uniqueCells[col[0][1]][1]] = -77
    
    return updatedGameMatrix

## Variable Declaration
top_left_edge_pix = np.array([[176,164,165,165,165,165],[172,156,157,157,157,157],
                               [172,156,157,157,157,157],[172,156,157,157,157,157],
                               [172,156,157,157,163,188],[172,156,157,157,172,235]])

bottom_right_edge_pix = np.array([[172, 238, 255, 255, 255, 223],[243, 252, 255, 255, 255, 223],
                                    [255, 255, 255, 255, 255, 223],[255, 255, 255, 255, 255, 223],
                                    [255, 255, 255, 255, 255, 223], [220, 220, 220, 220, 220, 206]])
     
cell_len = 24

## Program Start

ahk,win = runMinesweeper()
GameWindowEdges = getWindowCoordinates(win)

resetGame(win, ahk, GameWindowEdges)

# Get the Grid Information

screen_capture_array = getGameScreenshot(win, GameWindowEdges)      
gridEdges = getGridEdges(win, GameWindowEdges, screen_capture_array)
grid_array = CropScreenArray(screen_capture_array, gridEdges)
nrows = np.size(grid_array,0)//cell_len
ncols = np.size(grid_array,1)//cell_len
maxbombs = getInitialNumberofBombs() # Maximum number of bombs

wins = 0
for i in range(10):
    print("Game number {0}".format(i+1))
    wins += solvePuzzle(win, nrows, ncols, cell_len, GameWindowEdges, gridEdges, maxbombs)
    
print("{0} wins in {1} games".format(wins,i+1))
    