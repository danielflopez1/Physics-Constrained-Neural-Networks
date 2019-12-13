__docformat__ = "reStructuredText"

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

class Simulate:
    def __init__(self):
        self.run_physics = True
        self.screenSize= self.h = self.w = 800 #!check matrices when changing the screen size!
        self.screen = pygame.display.set_mode((self.screenSize, self.screenSize))
        #colors used for specific tasks
        self.screen.fill(THECOLORS["black"])
        self.notTrackedColor = THECOLORS['blue']
        self.boundaryColor = THECOLORS['green']
        self.trackedColor = THECOLORS['red']
        self.area1Color = THECOLORS['gray']
        self.space = pymunk.Space()                         #space in screen
        self.space.gravity = 0.0, -900.0
        self.clock = pygame.time.Clock()
        self.shapeCounter = 0
        self.running = True                                 #let the simulation continue
        #different shappes being added to space
        self.lines = []
        self.trackedShapes = []
        self.balls = []
        self.polys = []
        self.areas =[]
        self.fpsMultiplier = 10                          #frames multiplier for higher detail on the simulation
        self.fps = 120.0*self.fpsMultiplier
        self.seconds = 10                               #how many seconds you want to run.
        self.frameLimit = self.seconds * self.fps           #how many frames we want
        self.record = True                               #save frames
        self.getPictures = False                           #get video
        self.scenenum = 4                 #which simulation you would like to create
        self.nameFile = "ball_in_square_sider_v3"         #name the simulation (each needs a different name or it wont accept it
 
    # Small hacks to convert chipmunk physics to pygame coordinates
    def flipy(self,y):
        return -y + self.screenSize

    def flipyv(self, v):
        return int(v.x), int(-v.y + self.h)
    #create the object that will be tracked
    def createObservedBall(self,radius,mass,x,y,elasticity,impulseX,impulseY):
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        body.apply_impulse_at_local_point((impulseX, impulseY))
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = elasticity
        self.space.add(body, shape)
        self.trackedShapes.append(shape)
        self.shapeCounter = self.shapeCounter + 1
    #draw the object that will be tracked
    def drawObservedBall(self):
        for shape in self.trackedShapes:
            r = shape.radius
            v = shape.body.position
            rot = shape.body.rotation_vector
            p = int(v.x), int(self.flipy(v.y))
            p2 = Vec2d(rot.x, -rot.y) * r * 0.9
            pygame.draw.circle(self.screen, THECOLORS["red"], p, int(r))
    #create static lines (boundaries=
    def createBoundaries(self,line):
        X1,Y1,X2,Y2 = line[0],line[1],line[2],line[3]
        self.linep1 = (X1, Y1)
        self.linep2 = (X2, Y2)
        pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(self.space.static_body, (X1, self.flipy(Y1)), (X2, self.flipy(Y2)), 1.0)#change 1.0 for width
        shape.elasticity = 1.0
        self.space.add(shape)
        self.lines.append(shape)
        self.shapeCounter = self.shapeCounter + 1


    def setArea(self,x1,y1,x2,y2):
        self.areas.append([x1,y1,x2,y2])


    #def changeObject(self,object):


    def checkArea(self):
        for i,ball in enumerate(self.trackedShapes):
            pos = ball.body.position
            print i, int(pos[0]), int(pos[1])
            for area in self.areas:
                if(pygame.Rect(area).collidepoint(int(pos[0]), 800-int(pos[1]))):
                    print("COLLISSION")

        for i,ball in enumerate(self.balls):
            pos = ball.body.position
            print i, int(pos[0]), int(pos[1])
            for area in self.areas:
                if (pygame.Rect(area).collidepoint(int(pos[0]), 800-int(pos[1]))):
                    print("COLLISSION")

    def printArea(self):
        for area in self.areas:
            pygame.draw.rect(self.screen,self.area1Color,pygame.Rect(area))


    def drawLines(self):
        for line in self.lines:
            body = line.body
            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            p1 = pv1.x, self.flipy(pv1.y)
            p2 = pv2.x, self.flipy(pv2.y)
            pygame.draw.lines(self.screen, self.boundaryColor, False, [p1, p2])#add #[p1, p2],1)# width of border

    #create the ball object
    def createBalls(self,radius,mass,x,y,elasticity,impulseX,impulseY):
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        body.apply_impulse_at_local_point((impulseX, impulseY))
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = elasticity
        self.space.add(body, shape)
        self.balls.append(shape)
        self.shapeCounter = self.shapeCounter + 1
    #draw ball object
    def drawBalls(self):
        for ball in self.balls:
            r = ball.radius
            v = ball.body.position
            rot = ball.body.rotation_vector
            p = int(v.x), int(self.flipy(v.y))
            p2 = Vec2d(rot.x, -rot.y) * r * 0.9
            pygame.draw.circle(self.screen, self.notTrackedColor, p, int(r),1)
    #create box object
    def create_box(self, pos, size=20, mass=5.0, impulseX = 0, impulseY = 0):
        box_points = [(-size, -size), (-size, size), (size, size), (size, -size)]
        return self.create_poly(box_points, mass=mass, pos=pos, impulseX=impulseX,impulseY=impulseY)
    #create polygon (note that box is a poylgon) object
    def create_poly(self, points, mass=5.0, pos=(0, 0), impulseX=0,impulseY=0):
        moment = pymunk.moment_for_poly(mass, points, (0, 0))
        # moment = 1000
        body = pymunk.Body(mass, moment)
        body.apply_impulse_at_local_point((impulseX,impulseY))
        body.position = pos[0],pos[1]
        shape = pymunk.Poly(body, points)
        shape.elasticity = 1
        self.space.add(body, shape)
        self.polys.append(shape)
        self.shapeCounter = self.shapeCounter + 1
    #draw box and polygon objects
    def draw_poly(self):
        for poly in self.polys:
            body = poly.body
            ps = [p.rotated(body.angle) + body.position for p in poly.get_vertices()]
            ps.append(ps[0])
            ps = list(map(self.flipyv, ps))
            pygame.draw.lines(self.screen, self.notTrackedColor, False, ps)
    #createSlowingArea
    def create_Area(self,lines):
        pygame.draw.lines(self.screen, self.area1Color,False,lines)

    #get the whole size of the screen as an array of RGB
    def getScreen(self):
        ascreen = self.screen
        string_image = pygame.image.tostring(pygame.transform.flip(ascreen,False,True),'RGB',True)
        temp_surf = pygame.image.fromstring(string_image, (self.w, self.h), 'RGB')
        tmp_arr = pygame.surfarray.array3d(temp_surf).astype(int)
        return tmp_arr

    def printArray(self,arr,w,h,a):
        for x in range(0,h):
            print a,x,
            for y in range(0,w):
                print arr[y,x],
            print()
    #create always the same "random numbers"
    random.seed(123)
    #scenarios|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    #projectile Motion
    def scenario1(self):
        #create the Red ball that is being tracked
        self.createObservedBall(25, 1, 60, 4800, 1.0, 500, 250)
        # set how many frames we want
        #self.frameLimit = self.seconds * self.fps

    #ball in a square
    def scenario2(self):
        self.createObservedBall(25, 1, 350, 450, 1.0, 700, -600)
        # Create Initial Lines
        lines = [[750, 750, 750, 50], [50, 50, 50, 750], [750, 750, 50, 750], [50, 50, 750, 50]]
        for line in lines:
            self.createBoundaries(line)

    #tracked ball and other ball in square
    def scenario3(self):
        speedx = 400
        speedy = 100
        self.createObservedBall(25, 1, 150, 650, 1.0, speedx, speedy)
        # Create Initial Lines
        lines = [[750, 750, 750, 50], [50, 50, 50, 750], [750, 750, 50, 750], [50, 50, 750, 50]]
        for line in lines:
            self.createBoundaries(line)
        #create non tracked balls
        self.createBalls(25, 1, 550, 650, 1, -speedx, speedy)

    #tracked ball and 10 balls in a square with modifiable barrier size
    def scenario4(self):
        size = 0
        self.createObservedBall(25, 1, 720, 400, 1.0, 0, 0)
        # Create Initial Lines

        lines = [[750-size, 750-size, 750-size, 50+size], [50+size, 50+size, 50+size, 750-size],
                 [750-size, 750-size, 50+size, 750-size], [50+size, 50+size, 750-size, 50+size]]
        for line in lines:
            self.createBoundaries(line)
        #create non tracked balls
        numBalls = 0
        for x in range(numBalls):
            self.createBalls(25, 1, 80+size + divmod(x,10)[1] * 50, 90+size + divmod(x,10)[0] * 50, 1, random.randint(-1000,1000), random.randint(-1000,1000))

    #create tracked ball with box objects in a square
    def scenario5(self):
        self.createObservedBall(25, 1, 150, 150, 1.0, 400, -100)
        # Create Initial Lines
        lines = [[750, 750, 750, 50], [50, 50, 50, 750], [750, 750, 50, 750], [50, 50, 750, 50]]
        for line in lines:
            self.createBoundaries(line)
        h = 5
        for y in range(1, h + 1):
            # for x in range(1, y):
            x = 50
            s = 20
            p = Vec2d(300, 40) + Vec2d(0, x + y * s * 2)
            self.create_box(p, size=s, mass=1,impulseX=random.randint(-500,500),impulseY=random.randint(-500,500))

    #create tracked ball with boxes
    def scenario6(self):
        self.createObservedBall(25, 1, 150, 150, 1.0, 400, -100)
        # Create Initial Lines
        lines = [[750, 750, 750, 50], [50, 50, 50, 750], [750, 750, 50, 750], [50, 50, 750, 50]]
        for line in lines:
            self.createBoundaries(line)
        h = 1
        sizes = 20
        for y in range(1, h + 1):
            x = 50
            s = 20
            p = Vec2d(100, 100) + Vec2d(x + y * s * 4, 0)
            verts = [(random.randint(-sizes,0), random.randint(0,sizes)), (random.randint(0,sizes), random.randint(-sizes,0)),
                     (random.randint(-sizes,0), random.randint(0,sizes)), (random.randint(0,sizes), random.randint(-sizes,0))]
            self.create_poly(verts, 1, p,-400,random.randint(-500,500))
        self.seconds = 50

    #create tracked ball with polygons
    def scenario7(self):
        self.createObservedBall(25, 1, 150, 150, 1.0, 400, -100)
        # Create Initial Lines
        lines = [[750, 750, 750, 50], [50, 50, 50, 750], [750, 750, 50, 750], [50, 50, 750, 50]]
        for line in lines:
            self.createBoundaries(line)
        h = 5
        sizes = 27
        for y in range(1, h + 1):
            x = 50
            s = 20
            p = Vec2d(100, 100) + Vec2d(x + y * s * 4, 0)
            verts = [(random.randint(-sizes,0), random.randint(0,sizes)), (random.randint(0,sizes), random.randint(-sizes,0)),
                     (random.randint(-sizes,0), random.randint(0,sizes)), (random.randint(0,sizes), random.randint(-sizes,0))]
            self.create_poly(verts, 1, p,random.randint(-500,500),random.randint(-500,500))
        self.seconds = 50
    #create tracked ball with modifiable polygon and 10 balls
    def scenario8(self):
        self.createObservedBall(25, 1, 500, 500, 1.0, 600, -400)
        # Create Initial Lines
        size = 0
        lines = [[65+size, 65+size, 500-size, 50+size], [500-size, 50+size, 750-size, 350+size], [750-size, 350+size, 650-size, 650-size],
                 [650-size, 650-size, 500-size, 750-size],[500-size, 750-size, 50+size, 500-size],[65+size, 65+size, 50+size, 500-size]]
        for line in lines:
            self.createBoundaries(line)
        numBalls = 3
        for x in range(numBalls):
            self.createBalls(25, 1, 80+size + divmod(x, 10)[1] * 50, 290+size + divmod(x, 10)[0] * 50, 1,
                             random.randint(-800, 1000), random.randint(-1000, 1000))



    # create observed ball and 10 balls and simpler polygon for testing
    def scenario9(self):
        self.createObservedBall(25, 1, 100, 500, 1.0, 400, -100)
        # Create Initial Lines
        lines = [[65, 65, 500, 50], [500, 50, 450, 350], [450, 350, 650, 650], [650, 650, 500, 750],[500, 750, 50, 500],[65, 65, 50, 500]]
        for line in lines:
            self.createBoundaries(line)
        numBalls = 3
        for x in range(numBalls):
            self.createBalls(25, 1, 100 + x * 50, 400 + x * 50, 1, random.randint(-800, 800), random.randint(-800, 800))



    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    def main(self):
        pygame.init()

        #python's switch case (add scenarios here):
        if self.scenenum == 1:
            self.scenario1()
        elif self.scenenum ==2:
            self.scenario2()
        elif self.scenenum == 3:
            self.scenario3()
        elif self.scenenum == 4:
            self.scenario4()
        elif self.scenenum == 5:
            self.scenario5()
        elif self.scenenum == 6:
            self.scenario6()
        elif self.scenenum == 7:
            self.scenario7()
        elif self.scenenum == 8:
            self.scenario8()
        elif self.scenenum == 9:
            self.scenario9()
        elif self.scenenum == 10:
            self.scenario10()
        elif self.scenenum == 11:
            self.scenario11()
        elif self.scenenum == 12:
            self.scenario12()
        a = 0
        self.frameLimit = self.seconds * self.fps/self.fpsMultiplier
        if self.record:
            sqrname = self.nameFile+".h5"
            mainname = "full_"+self.nameFile+".h5"
            propname = "xyrot_"+self.nameFile+".h5"
            #squareMatrix = h5py.File(sqrname).create_dataset("matrices", shape=(self.frameLimit+1, 200, 200, 5))
            mainMatrix = h5py.File(mainname).create_dataset("matrices", shape=(self.frameLimit+1, self.w, self.h, 3))
            ballProperties = h5py.File(propname).create_dataset("matrices", shape=(self.frameLimit+1, 3))

        if self.getPictures:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            video = cv2.VideoWriter("out_" + self.nameFile[:-3] + ".avi", -1, 60, (self.w, self.h))

        while self.running:
            #file = open("matrix" + str(a)+".txt","w")
            if(divmod(a,self.fpsMultiplier)[0]>=self.frameLimit and self.run_physics): # Make it elastic so no energy i s lots      Add other shapes     Track one ball
                self.running = False
            #pygame.image.save(self.screen,"circledPhoto"+str(a)+".jpg")

            x1, y1, x2, y2 = str(self.space.shapes[0].cache_bb())[3:-1].split(',')
            space = 25


            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_p:
                    print(x1,y1)
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    self.run_physics = not self.run_physics

            ### Update physics #Change for fps
            if self.run_physics:
                dt = 1.0 / self.fps
                for x in range(1):
                    self.space.step(dt)

            # Display some text
            font = pygame.font.Font(None, 16)
            text = str(a)
            y = 5
            for line in text.splitlines():
                text = font.render(line, 1, THECOLORS["black"])
                self.screen.blit(text, (5, y))
                y += 10

            #self.space.debug_draw(self.draw_options)
            self.screen.fill(THECOLORS["white"])

            ### Draw polygons
            self.draw_poly()
            self.drawObservedBall()
            self.drawBalls()
            # Draw the lines
            self.drawLines()

            #Main Ball to be followed:
            #pygame.draw.circle(self.screen, THECOLORS["red"], (int(float(x1)) + 25, self.flipy(int(float(y1))) - 25), 25)
            #pygame.draw.circle(self.screen, THECOLORS["blue"], (int(float(x1)) + 25, self.flipy(int(float(y1))) - 25), 5) #Center
            #pygame.draw.rect(self.screen, THECOLORS["red"], (float(x1)-25,self.flipy(float(y2))-25, 100,100), 1) # get the rectangle around the main circle
            #self.setArea(300,400,100,100)
            #self.checkArea()
            #self.printArea()
            #record specifics of the game
            div,res = divmod(a, self.fpsMultiplier)
            ballarr = [int(float(x1)) + space] + [self.flipy(int(float(y1))) - space]
            #print(ballarr)
            if(res==0):
                if self.getPictures:
                    pygame.image.save(self.screen, self.nameFile+".jpg")
                    video.write(cv2.imread(self.nameFile+".jpg"))
                if self.record:
                    fullarr = self.getScreen()
                    mainMatrix[div] = fullarr
                    #arr, w, h = self.getRGBXY(x1, y1, x2, y2, space)
                    #squareMatrix[div] = arr
                    #self.printArray(arr,100,100,a)
                    roty = 0
                    for shape in self.trackedShapes:
                        roty = shape.body.rotation_vector[1]
                        centerline = str(float(x1) + space)+" "+ str(self.flipy(float(y1)) - space)+" "+str(roty)+"\n"
                    ballarr = np.zeros(shape=3)
                    ballarr = [int(float(x1)) + space] + [self.flipy(int(float(y1))) - space] + [roty]
                    #print(ballarr)
                    ballProperties[div] = ballarr
                ### Flip self.screen
            pygame.display.flip()

            self.clock.tick(30)
            pygame.display.set_caption("frame:"+str(div)+" fps: " + str(self.clock.get_fps()))

            a = a + 1
        if self.getPictures:
            video.release()

sim = Simulate()
sim.main()
