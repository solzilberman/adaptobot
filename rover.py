import random
import os
import pygame
import gsn.parse_gsn as gsn
import nn.ImgNet as nn
import torch
import numpy as np 
import math

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

## initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("rover")
clock = pygame.time.Clock()     ## For syncing the FPS


def rotateVector(v, theta):
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) # https://www.atqed.com/numpy-rotate-vector
    return rot.dot(v)



class Obstacle:
    def __init__(self, x, y, label, image):
        self.x = x
        self.y = y
        self.label = label
        self.image = image

    def draw(self):
        pygame_img = pygame.surfarray.make_surface(self.image[0].permute(*torch.arange(self.image[0].ndim - 1, -1, -1)).numpy())
        screen.blit(pygame_img, (self.x, self.y))

class Obstacles:
    def __init__(self):
        self.data = list(iter(nn._get_test_data()) )
        self.obstacles = []
        for img, label in self.data:
            self.obstacles.append(Obstacle(random.randint(0, WIDTH-20), random.randint(0, HEIGHT-20), label, img))
        self.obstacles[0].x = 30
        self.obstacles[0].y = 30

    def draw(self):
        for obstacle in self.obstacles:
            obstacle.draw()
        
        

# create a rover class which is a triangle 
class Rover():
    def __init__(self, waypoints, obstacles):
        # general params
        self.x = WIDTH/2
        self.y = HEIGHT/2
        self.width = 40
        self.height = 40
        self.speedx = 0
        self.speedy = 0
        self.view_range = 55
        # surface params
        # self.surface = pygame.image.load("assets/rover.png")
        # self.surface = pygame.transform.scale(self.surface, (self.width, self.height))
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.rect = pygame.Rect(0,0, self.width, self.height)
        self.surface.fill(GREEN)
        # battery params
        self.battery_level = 1000
        self.battery_delta = 10
        # mape-K params
        self.MODE = "drive"
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.gsnTree = gsn.GSNModel('gsn/gsn.xml')
        self.shouldUpdateDirection = True
        self.isUtilityViolated = False
        self.planFuncs = []
        self.model = nn._load_model()
        self.CvEnabled = False
        self.debug = True
        self.pred_debug_logs = "None"
        
    
    def update(self):
        pass
    
    def move(self):
        self.x += self.speedx
        self.y += self.speedy
        self.rect.center = (self.x, self.y)
    
    def rotate(self):
        ang = math.atan2(self.speedy,self.speedx)*(180 / math.pi)
        return pygame.transform.rotate(self.surface, ang)
    
    def draw_view_range(self):
        # get top left and top right of the triangle
        adj = np.array([self.speedx/self.get_speed(), self.speedy/self.get_speed()])*self.view_range
        theta = np.pi/6
        adj1 = rotateVector(adj, theta)
        adj2 = rotateVector(adj, -theta)
        pygame.draw.line(screen, WHITE, (self.x, self.y), (self.x + adj2[0], self.y + adj2[1]), 5)
        pygame.draw.line(screen, WHITE, (self.x, self.y), (self.x + adj1[0], self.y + adj1[1]), 5)
        pygame.draw.line(screen, WHITE, (self.x + adj1[0], self.y + adj1[1]), (self.x + adj2[0], self.y + adj2[1]), 5)

    def draw(self):
        # pygame.draw.rect(screen, GREEN, self.rect)
        rot_img = self.rotate()
        temp_rect = self.surface.get_rect(center = (self.rect.center ))
        screen.blit(rot_img, temp_rect)
        self.draw_view_range()
        if self.MODE == "charge":
            battery_surface = pygame.image.load("assets/solar.png")
            battery_surface = pygame.transform.scale(battery_surface, (self.width, self.height))
            battery_rect = battery_surface.get_rect(center = (self.rect.center ))
            screen.blit(battery_surface, battery_rect)
        
        
        
    def print_debug_logs(self):
        os.system('clear')
        print(f"current waypoint: {self.waypoints.current_waypoint_id}")
        print(f"distanct to waypoint: {self.get_distance_to_current_waypoint()}")
        print(f"battery level: {self.battery_level}")
        print(f"battery utility func: {self.evaluate_battery_utility_func()}")
        print(f"battery utility violated: {self.isUtilityViolated}")
        print(f"MODE: {self.MODE}")
        print(f"currently speed: {self.speedx, self.speedy}")
        print(f"CV Enabled: {self.CvEnabled}")
        print(f"pred_debug logs: {self.pred_debug_logs}")


    def update_direction_to_waypoint(self, waypoints):
        waypoint = waypoints.get_current_waypoint()
        xdir = waypoint.x - self.x
        ydir = waypoint.y - self.y
        delta = 10
        self.speedx = xdir/DIAG*delta
        self.speedy = ydir/DIAG*delta

    def reset_battery_level(self):
        self.battery_level = 1000

    def update_battery_level(self):
        self.battery_level -= self.battery_delta
    
    def get_speed(self):
        if self.speedx == 0 and self.speedy == 0:
            return 0
        return (self.speedx**2 + self.speedy**2)**0.5
    
    def make_prediction(self, obstacle):
        outputs = self.model(obstacle.image)
        _, predicted = torch.max(outputs, 1)
        self.pred_debug_logs = f"predicted: {max(predicted)}, actual: {max(obstacle.label)}"
        maxlabel = max(obstacle.label)
        maxpred = max(predicted)
        if maxlabel == maxpred:
            self.pred_debug_logs = f"CORRECT"
            return True
        return(max(obstacle.label), max(predicted))

    def check_obstacle_in_radar(self):
        obstacle = self.obstacles.obstacles[0]
        vec_rover_to_obstacle = np.array([obstacle.x - self.x, obstacle.y - self.y])
        vec_rover_to_obstacle = vec_rover_to_obstacle/np.linalg.norm(vec_rover_to_obstacle)
        angle_from_speed = np.cos(np.dot(vec_rover_to_obstacle, np.array([self.speedx, self.speedy])))
        dist_to_obstacle = np.linalg.norm([obstacle.x - self.x, obstacle.y - self.y])
        if angle_from_speed <= np.pi/6 and dist_to_obstacle <= self.view_range:
            self.CvEnabled = True
        else:
            self.CvEnabled = False


    def evaluate_battery_utility_func(self):
        batteryFunc = self.gsnTree.utility_funcs[0]
        batteryLevel = self.battery_level
        batteryNeeded = (self.get_distance_to_current_waypoint())
        return batteryFunc.func(batteryLevel, batteryNeeded)
    
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
    
    def adaptation_tactic_to_battery_utility_violation(self):
        # adaptation tactic (TODO: should come from KB)
        # set to charge mode
        # rover will switch to drive mode on its own once battery level is high enough
        self.MODE = "charge"

    def evaluate_utility_funcs(self): # this should request ROS topics but here it will just request hardcoded internal utility funcs
        self.isUtilityViolated = False # reset utility violations
        # evaluate battery utility func
        self.isUtilityViolated = not self.evaluate_battery_utility_func()

    def _monitor(self):
        self.check_rover_reaches_waypoint()
        self.evaluate_utility_funcs()
        if self.debug:
            self.print_debug_logs()
        self.check_obstacle_in_radar()

    def _analyze(self):
        if self.shouldUpdateDirection:
            self.planFuncs.append("self.update_direction_to_waypoint(self.waypoints)")
            self.planFuncs.append("self.shouldUpdateDirection = False")
        if self.isUtilityViolated:
            self.planFuncs.append("self.adaptation_tactic_to_battery_utility_violation()")
            self.planFuncs.append("self.isUtilityViolated = False")
        if self.CvEnabled:
            self.planFuncs.append("self.make_prediction(self.obstacles.obstacles[0])")
            self.planFuncs.append("self.cvEnabled = False")
        self.planFuncs.append("self.update()")
    
    def _plan(self):
        # will plan new direction to waypoint
        pass
    
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
    def __init__(self,x,y, radius=5):
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
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

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


waypoints = [
    Waypoint(30,30),
    Waypoint(WIDTH-30,30),
    Waypoint(30,HEIGHT-30),
    Waypoint(WIDTH-30,HEIGHT-30),
]
for i in range(3):
    waypoints.append(Waypoint(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
wps = Waypoints(waypoints)

# obstacles = [
#     Obstacle(100,100, 50)
# ]
obs = Obstacles()

r = Rover(wps,obs)
r.update_direction_to_waypoint(wps)

bgImg = pygame.image.load("assets/bg.jpg")
bgImg = pygame.transform.scale(bgImg, (WIDTH, HEIGHT))



## Game loop
running = True
while running:
    clock.tick(FPS)     
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    screen.blit(bgImg, (0,0))
    obs.obstacles[0].draw()
    r.mape_loop()
    wps.draw()
    pygame.display.flip()       

pygame.quit()