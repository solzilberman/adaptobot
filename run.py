# import random
import os
import pygame
# import gsn.gsn as gsn
# import nn.ImgNet as nn
# import torch
# import numpy as np 
# import math
# from kb import KnowledgeBase
from rover import *
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

## initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("rover")
clock = pygame.time.Clock()



waypoints = [
    Waypoint(30+random.randint(10,50),30+random.randint(-10,10), screen),
    Waypoint(WIDTH-30+random.randint(-10,10),30, screen),
    Waypoint(30+ random.randint(10,50),HEIGHT-30, screen),
    Waypoint(WIDTH-30,HEIGHT-30+random.randint(-10,10), screen),
]
# for i in range(3):
#     waypoints.append(Waypoint(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
wps = Waypoints(waypoints)

# obstacles = [
#     Obstacle(100,100, 50)
# ]
obs = Obstacles(screen)

r = Rover(wps,obs,screen)
r.update_direction_to_waypoint()

kbs = KnowledgeBase('gsn/gsn_2.xml', 'model/ImgNet.pth')
sas = SAS(kbs, r)

bgImg = pygame.image.load("assets/tex.jpg")
bgImg = pygame.transform.scale(bgImg, (WIDTH, HEIGHT))

def key_callback(event):
    # arrow keey move rover in that direction
    if event.key == pygame.K_LEFT:
        r.move_manual(-1,0)
    elif event.key == pygame.K_RIGHT:
        r.move_manual(1,0)
    elif event.key == pygame.K_UP:
        r.move_manual(0,-1)
    elif event.key == pygame.K_DOWN:
        r.move_manual(0,1)

## Game loop
running = True
while running:
    clock.tick(FPS)     
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            key_callback(event)

    screen.fill(BLACK)
    screen.blit(bgImg, (0,0))
    sas.rover.obstacles.draw()
    sas.rover.waypoints.draw()
    sas._mape_loop()
    # r.obstacles.draw()
    # r.waypoints.draw()
    # r.mape_loop()
    pygame.display.flip()       

pygame.quit()