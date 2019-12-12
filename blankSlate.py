import random
import numpy as np
import h5py
import os

class changeFrame:
    def __init__(self,size,imagefilepath):
        self.size = size
        self.file = h5py.File(imagefilepath)
        self.matrixShape = self.file['matrices'].shape

    def offsetx(self,x):
        return int(x-self.size/2)
    def offsety(self,y):
        return int(y-self.size/2)

    def boundCheck(self,x,y):
        if x <0 or x>self.matrixShape[1]-1 or y<0 or y >self.matrixShape[1]-1:
            return False
        else:
            return True

    def circle(self, arr,radius, xoffset,yoffset):
        x, y = 0, radius
        arr = self.plotCircle(arr, x, y, radius, xoffset,yoffset)
        return arr

    def symmetry_points(self,arr, x, y, radius,xoffset,yoffset):
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

    def plotCircle(self,arr, x, y, radius, xoffset,yoffset):
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

    def printArray(self,arr,w,h,a):
        for x in range(0,h):
            print a,x,
            for y in range(0,w):
                print arr[x][y],
            print()
    def printLayout(self,arr,a):
        # print a
        # print("000,000",arr[0,0][3],arr[0,0][4])
        # print("800,000",arr[0,w-1][3],arr[0,w-1][4])
        print(a," 400,400 ",
              arr[400,400])  # arr[int(w/2),int(h/2)][2],arr[int(w/2),int(h/2)][3],arr[int(w/2),int(h/2)][4])
        # print("000,800",arr[h-1,0][3],arr[h-1,0][4])
        # print("800,800 ", arr[w-1,h-1][3], arr[w-1,h-1][4])

    def output(self,i, realx,realy):
        frame = self.file['matrices'][i]
        #self.printArray(frame,800,800,"F")
        arr = np.zeros((self.size, self.size, 5)).astype(int)
        test = np.zeros((3,3))
        test[0] = [0.0, 255.0, 0.0]
        test[1] = [0.0,0.0,255.0]
        test[2] = [255.0,0.0,0.0]

        xoffset = self.offsetx(realy)
        yoffset = self.offsety(realx)
        #print(i,"Xoff",xoffset,xoffset+self.size," Yoff",yoffset,yoffset+self.size," Real",realx, realy, self.size)
        for x in range(xoffset, xoffset+self.size):
            for y in range(yoffset, yoffset+self.size):
                data = frame[y][x]
                if (np.array_equal(test[2], data)): #np.array_equal(test[0], data)):
                    #print(data, x, y, i)
                    frame[x][y] = [255, 255, 255]
                if self.boundCheck(x,y):
                    arr[x-xoffset][y-yoffset]= [int(data[0]), int(data[1]), int(data[2]),y, x]
                else:
                    arr[x-xoffset][y-yoffset] = [255,255,255,y, x]
        x0 = y0 = int(self.size / 4)
        arr = self.circle(arr,25,x0,y0)
        #self.printArray(arr,self.size,self.size,i)
        #self.printLayout(arr,i)
        return arr

''' 
size = 100
img = changeFrame(size, 'full_3ball_in_concave_polygon_v3.h5')
img.output(80,163,524)
'''