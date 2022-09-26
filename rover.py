import random
import os
import pygame
import gsn.parse_gsn as gsn
os.environ["SDL_VIDEODRIVER"] = "x11"

WIDTH = 500
HEIGHT = 500
DIAG = (WIDTH**2 + HEIGHT**2)**0.5
FPS = 30

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

# create a rover class which is a triangle 
class Rover():
    def __init__(self, waypoints):
        self.x = WIDTH/2
        self.y = HEIGHT/2
        self.width = 20
        self.height = 20
        self.speedx = 0
        self.speedy = 0
        self.rect = pygame.Rect(0,0, self.width, self.height)
        self.rect.center = (self.x, self.y)
        self.battery_level = 1000
        self.battery_delta = 10
        self.MODE = "drive"
        self.waypoints = waypoints
        self.gsnTree = gsn.GSNModel('gsn/gsn.xml')
        self.shouldUpdateDirection = True
        self.isUtilityViolated = False
        self.planFuncs = []
    
    def update(self):
        pass
    
    def move(self):
        self.x += self.speedx
        self.y += self.speedy
        self.rect.center = (self.x, self.y)
    
    def draw(self):
        pygame.draw.rect(screen, GREEN, self.rect)

    def print_debug_logs(self):
        os.system('clear')
        print(f"current waypoint: {self.waypoints.current_waypoint_id}")
        print(f"distanct to waypoint: {self.get_distance_to_current_waypoint()}")
        print(f"battery level: {self.battery_level}")
        print(f"battery utility func: {self.evaluate_battery_utility_func()}")
        print(f"battery utility violated: {self.isUtilityViolated}")
        print(f"MODE: {self.MODE}")
        print(f"currently speed: {self.speedx, self.speedy}")


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
        self.print_debug_logs()

    def _analyze(self):
        if self.shouldUpdateDirection:
            self.planFuncs.append("self.update_direction_to_waypoint(self.waypoints)")
            self.planFuncs.append("self.shouldUpdateDirection = False")
        if self.isUtilityViolated:
            self.planFuncs.append("self.adaptation_tactic_to_battery_utility_violation()")
            self.planFuncs.append("self.isUtilityViolated = False")
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

r = Rover(wps)
r.update_direction_to_waypoint(wps)



## Game loop
running = True
while running:
    clock.tick(FPS)     
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    r.mape_loop()
    wps.draw()
    pygame.display.flip()       

pygame.quit()