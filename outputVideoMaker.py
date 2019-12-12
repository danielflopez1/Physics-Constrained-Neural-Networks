__docformat__ = "reStructuredText"

import pygame, cProfile, pstats
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
import time


class outputVideo:
    def __init__(self,file,lines):
        self.run_physics = True
        self.screenSize = self.h = self.w = 800
        self.screen = pygame.display.set_mode((self.screenSize, self.screenSize))
        self.screen.fill(THECOLORS["black"])
        self.notTrackedColor = THECOLORS['blue']
        self.boundaryColor = THECOLORS['green']
        self.trackedColor = THECOLORS['red']
        self.clock = pygame.time.Clock()
        self.shapeCounter = 0
        self.running = True
        self.lines = lines
        self.trackedShapes = []
        self.balls = []
        self.polys = []
        self.mov_avg = []
        self.points = []
        self.fps = 120.0
        self.seconds = 3
        self.frameLimit = 50000
        self.getPictures = False
        self.videoName = "Time_marching_square_modified2"
        self.nameFile = file
        self.type = "csv"
        self.timer = 0.02
        self.skip = 0
        self.change =3
        self.trainLimit = self.change
        self.moving_average_mag = 3

    def flipy(self, y):
        """Small hack to convert chipmunk physics to pygame coordinates"""
        return -y + self.screenSize

    def flipyv(self, v):
        return int(v.x), int(-v.y + self.h)

    def drawObservedBall(self,x,y, radius):
        r = radius
        p = int(x), int(y)
        pygame.draw.circle(self.screen, self.trackedColor, p, int(r))

    def drawLines(self):
        for line in self.lines:
            p1 = line[0], line[1]
            p2 = line[2], line[3]
            pygame.draw.lines(self.screen, self.boundaryColor, False, [p1, p2])

    def drawBalls(self,x,y,radius):
        r = radius
        p = int(x), int(y)
        pygame.draw.circle(self.screen, self.notTrackedColor, p, int(r), 1)

    def drawBalls2(self,x,y,radius):
        r = radius
        p = int(x), int(y)
        pygame.draw.circle(self.screen, self.trackedColor, p, int(r), 1)

    def draw_poly(self):
        for poly in self.polys:
            body = poly.body
            ps = [p.rotated(body.angle) + body.position for p in poly.get_vertices()]
            ps.append(ps[0])
            ps = list(map(self.flipyv, ps))
            pygame.draw.lines(self.screen, self.notTrackedColor, False, ps)

    def draw_moving(self):
        for point in self.mov_avg:
            p1 = point[0], point[1]
            p2 = point[2], point[3]
            pygame.draw.lines(self.screen, self.trackedColor, False, [p1, p2])

    def printArray(self, arr, w, h, a):
        for x in range(0, h):
            print a, x,
            for y in range(0, w):
                print arr[x, y],
            print()


    def get_moving_averages(self,points):
        print(self.points)
        if(len(self.points)<self.moving_average_mag):
            self.points.append(points)
        else:
            avgs = np.mean(self.points,0)
            #print("Mean",avgs)
            '''
            x_dir = self.points[self.moving_average_mag-1][0] - self.points[0][0]
            y_dir = self.points[self.moving_average_mag-1][1] - self.points[0][1]
            #print("Max_points_dir",x_dir,y_dir)

            old_point = self.points[0]
            xdir = 0
            ydir = 0
            for point in self.points[1:]:
                xdir += point[0]-old_point[0]
                ydir += point[1]-old_point[1]
                #print(xdir,ydir)
                old_point = point
            mov_xdir = xdir/(self.moving_average_mag-1)
            mov_ydir = ydir /(self.moving_average_mag-1)
            print(mov_xdir,mov_ydir,avgs)
            '''
            self.mov_avg.append(np.append(self.points[0],[avgs]))
            self.draw_moving()
            self.points = self.points[1:]
            self.points.append(points)




    def getScreen(self):
        ascreen = self.screen
        string_image = pygame.image.tostring(pygame.transform.flip(ascreen,True,False),'RGB',True)
        temp_surf = pygame.image.fromstring(string_image, (self.w, self.h), 'RGB')
        tmp_arr = pygame.surfarray.array3d(temp_surf)
        return tmp_arr

    # create always the same "random numbers"
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    def main(self):
        pygame.init()
        if(self.type == "h5"):
            ballProperties = h5py.File(self.nameFile)
        if self.getPictures:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            video = cv2.VideoWriter("out_"+self.videoName + ".avi", -1, 60, (self.w, self.h))
        a =0
        predictedFile = open("predicted200.csv",'r')
        realFile = open("real.csv",'r')
        while(self.running):
            skipit = a<self.skip
            rline = realFile.readline()
            if (skipit):
                a = a+ 1
                print(a)
                continue
            rline =rline.replace('[','')
            rline =rline.replace(']','')
            splits = rline.split('.')
            rx = int(splits[0])
            ry = int(splits[1])
            if (a < self.trainLimit):
                self.drawBalls(rx, ry, 25)

            if a > self.trainLimit:
                pline = predictedFile.readline()
                # print("|",pline,"|")
                splits = pline.split(',')
                if (a > self.trainLimit):
                    self.timer = 0.2
                # print(splits, a)
                px = int(splits[0])
                py = int(splits[1])
                # self.drawBalls2(px, py, 25)
                self.get_moving_averages([px, py])
                # print("rline")

            # file = open("matrix" + str(a)+".txt","w")
            #a >= len(ballProperties['matrices'])-1 or
            if ( a >= self.frameLimit):  # Make it elastic so no energy i s lots      Add other shapes     Track one ball
                self.running = False

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    self.run_physics = not self.run_physics
            font = pygame.font.Font(None, 16)
            text = str(a)
            ys = 5
            for line in text.splitlines():
                text = font.render(line, 1, THECOLORS["black"])
                self.screen.blit(text, (5, ys))
                ys += 10
            if(self.skip==a):
                self.screen.fill(THECOLORS["white"])

            ### Draw polygons
            #if (a > 1):
                #self.drawObservedBall(rx,ry,25)
            self.drawLines()


            time.sleep(self.timer)

            # Main Ball to be followed:
            # pygame.draw.circle(self.screen, THECOLORS["red"], (int(float(x1)) + 25, self.flipy(int(float(y1))) - 25), 25)
            # pygame.draw.circle(self.screen, THECOLORS["blue"], (int(float(x1)) + 25, self.flipy(int(float(y1))) - 25), 5) #Center
            #if a > self.trainLimit:
            #    pygame.draw.rect(self.screen, THECOLORS["red"], (float(px)-50,float(py)-50, 100,100), 1) # get the rectangle around the main circle

            if self.getPictures:
                pygame.image.save(self.screen, self.nameFile + ".jpg")
                video.write(cv2.imread(self.nameFile + ".jpg"))
            ### Flip self.screen
            pygame.display.flip()
            self.clock.tick(30)
            pygame.display.set_caption("frame:" + str(a) + " fps: " + str(self.clock.get_fps()))
            a = a + 1
        if self.getPictures:
            video.release()


size = 0
lines = [[65 + size, 65 + size, 500 - size, 50 + size], [500 - size, 50 + size, 750 - size, 350 + size],
         [750 - size, 350 + size, 650 - size, 650 - size],
         [650 - size, 650 - size, 500 - size, 750 - size], [500 - size, 750 - size, 50 + size, 500 - size],
         [65 + size, 65 + size, 50 + size, 500 - size]]
#lines = [[750, 750, 750, 50], [50, 50, 50, 750], [750, 750, 50, 750], [50, 50, 750, 50]]
sim = outputVideo("time_marching_ball_on_polygon.h5",lines)
sim.main()

