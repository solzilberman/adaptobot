import random
import os
import pygame
import gsn.gsn as gsn
import nn.ImgNet as nn
import torch
import numpy as np 
import math
from kb import KnowledgeBase
from sas import SAS

os.environ["SDL_VIDEODRIVER"] = "x11"

WIDTH = 500
HEIGHT = 500
DIAG = (WIDTH**2 + HEIGHT**2)**0.5
FPS = 15

# Define Colors 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

def rotateVector(v, theta):
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) # https://www.atqed.com/numpy-rotate-vector
    return rot.dot(v)

class Obstacle:
    def __init__(self, x, y, label, image, screen):
        self.x = x
        self.y = y
        self.label = label
        self.image = image
        self.screen = screen

    def draw(self):
        pygame_img = pygame.surfarray.make_surface(self.image[0].permute(*torch.arange(self.image[0].ndim - 1, -1, -1)).numpy())
        self.screen.blit(pygame_img, (self.x, self.y))

class Obstacles:
    def __init__(self, screen):
        self.data = list(iter(nn._get_test_data()) )
        self.obstacles = []
        for img, label in self.data:
            self.obstacles.append(Obstacle(random.randint(0, WIDTH-20), random.randint(0, HEIGHT-20), label, img, screen))
        self.current_obstacle_index = 0
        self.obstacles[self.current_obstacle_index].x = 30
        self.obstacles[self.current_obstacle_index].y = 30

    def update_current_obstacle(self):
        self.current_obstacle_index += 1
        self.current_obstacle_index %= len(self.obstacles)

    def get_current_obstacle(self):
        return self.obstacles[self.current_obstacle_index]

    def draw(self):
        self.get_current_obstacle().draw()
        
        

# create a rover class which is a triangle 
class Rover():
    def __init__(self, waypoints, obstacles, screen):
        # general params
        self.screen = screen
        self.x = WIDTH/2
        self.y = HEIGHT/2
        self.width = 40
        self.height = 40
        self.speedx = 0
        self.speedy = 0
        self.delta = 5
        self.view_range = 55
        # surface params
        # self.surface = pygame.image.load("assets/rover.png")
        # self.surface = pygame.transform.scale(self.surface, (self.width, self.height))
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.rect = pygame.Rect(0,0, self.width, self.height)
        self.rect.center = (self.x, self.y)
        self.surface.fill(YELLOW)
        # battery params
        self.battery_level = 1000
        self.battery_delta = 10
        # mape-K params
        self.MODE = "drive"
        self.knowledge_base = KnowledgeBase('gsn/gsn_2.xml', 'model/ImgNet.pth')
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.gsnTree = self.knowledge_base.gsn
        self.shouldUpdateDirection = True
        self.model = self.knowledge_base.nn
        self.cv_threshold = 0.5
        self.CvDegraded = False
        # utility function params        
        self.isUtilityViolatedBattery = False
        self.isUtilityViolatedCV = False
        self.planFuncs = []
        self.debug = True
        self.pred_debug_logs = "None"
        
    
    def update(self):
        pass
    
    def move(self):
        self.x += self.speedx*self.delta
        self.y += self.speedy*self.delta
        self.rect.center = (self.x, self.y)
    
    def move_manual(self, dx, dy):
        self.x += dx
        self.y += dy
        self.rect.center = (self.x, self.y)
    
    def rotate(self):
        ang = np.arccos(np.dot([self.speedx, self.speedy], [-1,0]))*(180/np.pi)
        tmp = self.surface.copy()
        return pygame.transform.rotate(tmp, ang)
    
    def draw_view_range(self):
        # get top left and top right of the triangle
        adj = np.array([self.speedx, self.speedy])*self.view_range
        theta = np.pi/6
        adj1 = rotateVector(adj, theta)
        adj2 = rotateVector(adj, -theta)
        cx,cy = self.rect.center
        # pygame.draw.line(self.screen, WHITE, (cx, cy), (cx + adj2[0], cy + adj2[1]), 5)
        # pygame.draw.line(self.screen, WHITE, (cx, cy), (cx + adj1[0], cy + adj1[1]), 5)
        # pygame.draw.line(self.screen, WHITE, (cx + adj1[0], cy + adj1[1]), (cx + adj2[0], cy + adj2[1]), 5)
        # wp = self.waypoints.get_current_waypoint()
        # pygame.draw.line(self.screen, RED, (cx, cy), (wp.x,wp.y), 2)
        # obs = self.obstacles.get_current_obstacle()
        # pygame.draw.line(self.screen, RED, (cx, cy), (obs.x,obs.y), 2)


    def draw(self):
        # pygame.draw.rect(self.screen, GREEN, self.rect)
        rot_img = self.rotate()
        temp_rect = self.surface.get_rect(center = (self.rect.center ))
        self.screen.blit(rot_img, temp_rect)
        self.draw_view_range()
        if self.MODE == "charge":
            battery_surface = pygame.image.load("assets/solar.png")
            battery_surface = pygame.transform.scale(battery_surface, (self.width, self.height))
            battery_rect = battery_surface.get_rect(center = (self.rect.center ))
            self.screen.blit(battery_surface, battery_rect)
        
        
        
    def print_debug_logs(self):
        os.system('clear')
        print(f"current waypoint: {self.waypoints.current_waypoint_id}")
        print(f"distanct to waypoint: {self.get_distance_to_current_waypoint()}")
        print(f"battery level: {self.battery_level}")
        print(f"battery utility violated: {self.isUtilityViolatedBattery}")
        print(f"cv utility violated: {self.isUtilityViolatedCV}")
        print(f"MODE: {self.MODE}")
        print(f"currentl velocity: {self.speedx, self.speedy}")
        print(f"CV Degraded: {self.CvDegraded}")
        print(f"pred_debug logs: {self.pred_debug_logs}")
        obstacle = self.obstacles.get_current_obstacle()
        dist_to_obstacle = np.linalg.norm([obstacle.x - self.rect.center[0], obstacle.y - self.rect.center[1]])
        print(f"current obstacle dist: {dist_to_obstacle}")


    def update_direction_to_waypoint(self):
        waypoint = self.waypoints.get_current_waypoint()
        xdir = waypoint.x - self.x
        ydir = waypoint.y - self.y
        mag = math.sqrt(xdir**2 + ydir**2)
        delta = 10
        self.speedx = xdir/mag
        self.speedy = ydir/mag

    def reset_battery_level(self):
        self.battery_level = 1000

    def update_battery_level(self):
        self.battery_level -= self.battery_delta
    
    def get_speed(self):
        if self.speedx == 0 and self.speedy == 0:
            return 0
        return math.sqrt(self.speedx**2 + self.speedy**2)*self.delta

    def make_prediction(self, obstacle):
        outputs = self.model(obstacle.image)
        _, predicted = torch.max(outputs, 1)
        # calculate accuracy predicted vs labels
        correct = len(list(filter(lambda x: x[0] == x[1], zip(predicted, obstacle.label))))
        accuracy = correct/len(predicted)
        self.pred_debug_logs = f"Accuracy: {accuracy}"
        if accuracy > self.cv_threshold:
            return True, accuracy
        return False, accuracy

    def check_obstacle_in_radar(self):
        obstacle = self.obstacles.get_current_obstacle()
        vec_rover_to_obstacle = np.array([obstacle.x - self.rect.center[0], obstacle.y - self.rect.center[1]])
        vec_rover_to_obstacle = vec_rover_to_obstacle/np.linalg.norm(vec_rover_to_obstacle)
        angle_from_speed = np.cos(np.dot(vec_rover_to_obstacle, np.array([self.speedx, self.speedy])))
        dist_to_obstacle = np.linalg.norm([obstacle.x - self.rect.center[0], obstacle.y - self.rect.center[1]])
        if angle_from_speed <= np.pi/2 and abs(dist_to_obstacle-self.view_range) <= 20:
            b, acc = self.make_prediction(obstacle)
            if b:
                self.CvDegraded = False
                self.obstacles.update_current_obstacle()
            else:
                self.CvDegraded = True

    def charge_solar_panel(self):
        self.battery_level += self.battery_delta
        if self.battery_level > 1000:
            self.MODE = "drive"
    
    def update(self):
        if self.MODE == "drive":
            self.move()
            self.update_battery_level()
        elif self.MODE == "charge":
            self.charge_solar_panel()
        elif self.MODE == "manual":
            pass
        self.draw()
    
    def get_distance_to_current_waypoint(self):
        waypoint = self.waypoints.get_current_waypoint()
        return ((self.x - waypoint.x)**2 + (self.y - waypoint.y)**2)**0.5

    def check_rover_reaches_waypoint(self):
        waypoint = self.waypoints.get_current_waypoint()
        dist = self.get_distance_to_current_waypoint()
        if dist < waypoint.radius:
            self.waypoints.update_current_waypoint()
            self.shouldUpdateDirection = True
            if self.MODE == "manual":
                self.MODE = "drive"
    
    def evaluate_battery_utility_func(self):
        batteryFunc = self.gsnTree.utility_funcs[0]
        batteryLevel = self.battery_level
        batteryNeeded = (self.get_distance_to_current_waypoint())
        return not batteryFunc.func(batteryLevel, batteryNeeded)
    
    def get_battery_data(self):
        batteryFunc = self.gsnTree.utility_funcs[0]
        batteryLevel = self.battery_level
        batteryNeeded = (self.get_distance_to_current_waypoint())
        return  batteryLevel, batteryNeeded

    def get_cv_data(self):
        return self.CvDegraded

    def evaluate_cv_utility_func(self):
        return self.CvDegraded # TODO should be actual utility func 

    def adaptation_tactic_to_battery_utility_violation(self):
        # adaptation tactic (TODO: should come from KB)
        # set to charge mode
        # rover will switch to drive mode on its own once battery level is high enough
        self.MODE = "charge"
    
    def adaptation_tactic_to_cv_utility_violation(self):
        # adaptation tactic (TODO: should come from KB)
        # set to charge mode
        # rover will switch to drive mode on its own once battery level is high enough
        self.MODE = "manual"
        self.CvDegraded = False

    def evaluate_utility_funcs(self): 
        # this should request ROS topics but here it will just request hardcoded internal utility funcs
        self.isUtilityViolated = False # reset utility violations
        # evaluate battery utility func
        self.isUtilityViolatedBattery = self.evaluate_battery_utility_func()
        self.isUtilityViolatedCV = self.evaluate_cv_utility_func()

    def _monitor(self):
        if self.MODE == "drive": # only monitor obs when driving
            self.check_obstacle_in_radar()
        self.check_rover_reaches_waypoint()
        self.evaluate_utility_funcs()
        if self.debug:
            self.print_debug_logs()

    def _analyze(self):
        pass
    
    def _plan(self):
        if self.shouldUpdateDirection:
            self.planFuncs.append("self.update_direction_to_waypoint()")
            self.planFuncs.append("self.shouldUpdateDirection = False")
        if self.isUtilityViolatedBattery:
            self.planFuncs.append("self.adaptation_tactic_to_battery_utility_violation()")
            self.planFuncs.append("self.isUtilityViolatedBattery = False")
        if self.isUtilityViolatedCV:
            self.planFuncs.append("self.adaptation_tactic_to_cv_utility_violation()")
            self.planFuncs.append("self.isUtilityViolatedCV = False")
        self.planFuncs.append("self.update()")
    
    def _execute(self):
        for func in self.planFuncs:
            exec(func)
        self.planFuncs = []

    def mape_loop(self):
        self._monitor()
        self._analyze()
        self._plan()
        self._execute() 

class Waypoint():
    def __init__(self,x,y, screen, radius=5):
        self.screen = screen
        self.radius = radius
        self.x = x
        self.y = y
        self.speedx = 0
        self.speedy = 0
        self.color = RED
    
    def update(self):
        pass
    
    def move(self, x, y):
        self.speedx = x
        self.speedy = y
        self.rect.x += self.speedx
        self.rect.y += self.speedy
    
    def draw(self):
        pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius)

class Waypoints():
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint_id = 0
        self.get_current_waypoint().color = ORANGE

    def get_current_waypoint(self):
        return self.waypoints[self.current_waypoint_id]

    def update_current_waypoint(self):
        self.get_current_waypoint().color = RED
        self.current_waypoint_id = (self.current_waypoint_id + 1) % len(self.waypoints)
        self.get_current_waypoint().color = ORANGE

    def draw(self):
        for waypoint in self.waypoints:
            waypoint.draw()


# waypoints = [
#     Waypoint(30+random.randint(10,50),30+random.randint(-10,10)),
#     Waypoint(WIDTH-30+random.randint(-10,10),30),
#     Waypoint(30+ random.randint(10,50),HEIGHT-30),
#     Waypoint(WIDTH-30,HEIGHT-30+random.randint(-10,10)),
# ]
# # # for i in range(3):
# # #     waypoints.append(Waypoint(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
# # wps = Waypoints(waypoints)

# # # obstacles = [
# # #     Obstacle(100,100, 50)
# # # ]
# # obs = Obstacles()

# # r = Rover(wps,obs)
# # r.update_direction_to_waypoint()

# # kbs = KnowledgeBase('gsn/gsn_2.xml', 'model/ImgNet.pth')
# # sas = SAS(kbs, r)

# # bgImg = pygame.image.load("assets/tex.jpg")
# # bgImg = pygame.transform.scale(bgImg, (WIDTH, HEIGHT))

# # def key_callback(event):
# #     # arrow keey move rover in that direction
# #     if event.key == pygame.K_LEFT:
# #         r.move_manual(-1,0)
# #     elif event.key == pygame.K_RIGHT:
# #         r.move_manual(1,0)
# #     elif event.key == pygame.K_UP:
# #         r.move_manual(0,-1)
# #     elif event.key == pygame.K_DOWN:
# #         r.move_manual(0,1)

# # ## Game loop
# # running = True
# # while running:
# #     clock.tick(FPS)     
# #     for event in pygame.event.get():        
# #         if event.type == pygame.QUIT:
# #             running = False
# #         if event.type == pygame.KEYDOWN:
# #             key_callback(event)

# #     self.screen.fill(BLACK)
# #     self.screen.blit(bgImg, (0,0))
# #     sas.rover.obstacles.draw()
# #     sas.rover.waypoints.draw()
# #     sas._mape_loop()
# #     # r.obstacles.draw()
# #     # r.waypoints.draw()
# #     # r.mape_loop()
# #     pygame.display.flip()       

# # pygame.quit()