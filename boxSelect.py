import pygame,cProfile, pstats
from pygame.locals import *
from pygame.color import *
import random
import numpy as np
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
import pymunk.util as u
import h5py
import cv2
import os

class selectImage:
    def __init__(self,size,imagefilepath,xyfilepath):
        self.size = size
        self.file = h5py.File(imagefilepath)
        self.matrixShape = self.file['matrices'].shape
        self.xy = h5py.File(xyfilepath)

        under = xyfilepath.find("_")
        name = xyfilepath[under + 1:-3] + "_tbox" + str(size) + ".h5"
        #self.outfile = h5py.File(name).create_dataset("matrices", shape=(self.matrixShape[0], size, size, 5))

    def offsetx(self,x):
        return int(x-self.size/2)
    def offsety(self,y):
        return int(y-self.size/2)

    def boundCheck(self,x,y):
        if x <0 or x>self.matrixShape[1]-1 or y<0 or y >self.matrixShape[1]-1:
            return False
        else:
            return True

    def printArray(self,arr,w,h,a):
        for x in range(0,h):
            print a,x,
            for y in range(0,w):
                print arr[x,y],
            print()

    def printLayout(self,arr,a):
        # print a
        print("000,000",arr[0,0][3],arr[0,0][4])
        print("800,000",arr[0,w-1][3],arr[0,w-1][4])
        print(a," 400,400 ",
              arr[50,50])  # arr[int(w/2),int(h/2)][2],arr[int(w/2),int(h/2)][3],arr[int(w/2),int(h/2)][4])
        print("000,800",arr[h-1,0][3],arr[h-1,0][4])
        print("800,800 ", arr[w-1,h-1][3], arr[w-1,h-1][4])

    def output(self):
        for i,frame in enumerate(self.file['matrices']):
            arr = np.zeros((self.size, self.size, 5)).astype(int)
            realx,realy,realrot = self.xy['matrices'][i]
            print(realx,realy)
            xoffset = self.offsetx(realy)
            yoffset = self.offsety(realx)
            print(i,"Xoff",xoffset,xoffset+self.size," Yoff",yoffset,yoffset+self.size," Real",realx, realy, size)
            for x in xrange(xoffset,xoffset+self.size):
                for y in xrange(yoffset,yoffset+self.size):
                    if self.boundCheck(x,y):
                        data = frame[y][x]
                        arr[x-xoffset][y-yoffset]= [int(data[0]), int(data[1]), int(data[2]),y, x]
                    else:
                        arr[x-xoffset][y-yoffset] = [255,255,255,y, x]
            #print (realx,realy,realrot)
            self.printArray(arr,size,size,i)
            #self.printLayout(arr,i)
            #self.outfile[i] = arr
            raw_input()

'''
size = 100
img = selectImage(size, "full_ball_in_square_sider_v3.h5","xyrot_ball_in_square_sider_v3.h5")
img.output()
'''

