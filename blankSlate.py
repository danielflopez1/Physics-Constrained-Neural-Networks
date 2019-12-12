import random
import numpy as np
import h5py
import os

class changeFrame:
    def __init__(self,size,imagefilepath):
        self.size = size
        self.file = h5py.File(imagefilepath)            #get image coordinates of where the ball is going through
        self.matrixShape = self.file['matrices'].shape  #get all of the frames of the project

    def offsetx(self,x):                      #Generate offsets for centering the ball
        return int(x-self.size/2)
    def offsety(self,y):
        return int(y-self.size/2)

    def boundCheck(self,x,y):                   #check the frame slize has not left the picture
        if x <0 or x>self.matrixShape[1]-1 or y<0 or y >self.matrixShape[1]-1:
            return False
        else:
            return True

    def symmetry_points(self,arr, x, y, radius,xoffset,yoffset): #Generate a red semicircles of specific size and place
        red = np.zeros((1, 3))
        for i in range(-x + radius,x + radius):
            red[0] = [250, 0, 0]
            arr[-y + radius+xoffset][i+yoffset][:3] = red[0]

        for i in range(-y + radius,y + radius):
            red[0] = [255.0, 0.0, 0.0]
            arr[-x + radius+xoffset][i+yoffset][:3] = red[0]

        for i in range(-y + radius,y + radius):
            red[0] = [255.0, 0.0, 0.0]
            arr[x + radius+xoffset][i+yoffset][:3] = red[0]

        for i in range(-x + radius,x + radius):
            red[0] = [255.0, 0.0, 0.0]
            arr[y + radius+xoffset][i+yoffset][:3] = red[0]

    def plotCircle(self,arr, x, y, radius, xoffset,yoffset):  #Plot full circle
        d = 5 / 4.0 - radius
        self.symmetry_points(arr, x, y, radius,xoffset,yoffset)
        while x < y:
            if d < 0:
                x += 1
                d += 2 * x + 1
            else:
                x += 1
                y -= 1
                d += 2 * (x - y) + 1
            self.symmetry_points(arr, x, y, radius,xoffset,yoffset)
        return arr

    def circle(self, arr,radius, xoffset,yoffset):  #plot circle in new frame
        x, y = 0, radius
        arr = self.plotCircle(arr, x, y, radius, xoffset,yoffset)
        return arr
    
    def printArray(self,arr,w,h,a): #Print image
        for x in range(0,h):
            print a,x,
            for y in range(0,w):
                print arr[x][y],
            print()
    def printLayout(self,arr,a): #get center of the image
        print(a," 400,400 ",
              arr[400,400])  
        
     # Generate new frames of specific size = self.size with ball of radius 25pxl using background images
    def output(self,i, realx,realy):
        frame = self.file['matrices'][i]
        #self.printArray(frame,800,800,"F")
        arr = np.zeros((self.size, self.size, 5)).astype(int)
        test = np.zeros((3,3))
        test[0] = [0.0, 255.0, 0.0]
        test[1] = [0.0,0.0,255.0]
        test[2] = [255.0,0.0,0.0]
        #Add more colors here for future tests
        xoffset = self.offsetx(realy)
        yoffset = self.offsety(realx)
        for x in range(xoffset, xoffset+self.size):         #colors such as blue of obstacles and green of boundaries are not eliminated
            for y in range(yoffset, yoffset+self.size):     #this section can be modified for later colors
                data = frame[y][x]
                if (np.array_equal(test[2], data)):         #If it is red make it white 
                    frame[x][y] = [255, 255, 255]
                if self.boundCheck(x,y):            #check the bounds
                    arr[x-xoffset][y-yoffset]= [int(data[0]), int(data[1]), int(data[2]),y, x]
                else:
                    arr[x-xoffset][y-yoffset] = [255,255,255,y, x]  #make it white if other colors in there
        x0 = y0 = int(self.size / 4)
        arr = self.circle(arr,25,x0,y0)         #Generate the red ball and add it to the frame
        return arr

''' 
#Example
size = 100
img = changeFrame(size, 'full_3ball_in_concave_polygon_v3.h5')
img.output(80,163,524)
'''
