# import random
import os
import pygame
from rover import *
from sas import SAS
import rec 
import pygame_gui

os.environ["SDL_VIDEODRIVER"] = "x11"

WIDTH = 500
HEIGHT = 500
GUI_WIDTH = 0
DIAG = (WIDTH**2 + HEIGHT**2) ** 0.5
FPS = 15

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH+GUI_WIDTH, HEIGHT))
pygame.display.set_caption("rover")
clock = pygame.time.Clock()


waypoints = [
    Waypoint(30 + random.randint(10, 50), 30 + random.randint(-10, 10), screen),
    Waypoint(WIDTH - 30 + random.randint(-10, 10), 30, screen),
    Waypoint(30 + random.randint(10, 50), HEIGHT - 30, screen),
    Waypoint(WIDTH - 30, HEIGHT - 30 + random.randint(-10, 10), screen),
]
# for i in range(3):
#     waypoints.append(Waypoint(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
wps = Waypoints(waypoints)

obs = Obstacles(screen)
kbs = KnowledgeBase("gsn/gsn_2.xml", "model/ImgNet.pth")
r = Rover(wps, obs, kbs.nn, screen)
r.update_direction_to_waypoint()

# init adaptive system
sas = SAS(kbs, r)

bgImg = pygame.image.load("assets/tex.jpg")
bgImg = pygame.transform.scale(bgImg, (WIDTH, HEIGHT))

def key_callback(event):
    # arrow keey move rover in that direction
    if event.key == pygame.K_LEFT:
        r.move_manual(-1, 0)
    elif event.key == pygame.K_RIGHT:
        r.move_manual(1, 0)
    elif event.key == pygame.K_UP:
        r.move_manual(0, -1)
    elif event.key == pygame.K_DOWN:
        r.move_manual(0, 1)


manager = pygame_gui.UIManager((WIDTH+GUI_WIDTH, HEIGHT))
hack_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((550, 275), (200, 50)),
                                             text='hack rover',
                                             manager=manager)

## Game loop
fc = 0
save = False
running = True
pygame_screen_recorder = rec.pygame_screen_recorder("./movie.gif")
while running:
    time_delta = clock.tick(60)/1000.0
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            key_callback(event)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == hack_button:
                print('Hello World!')
        manager.process_events(event)

    screen.fill(BLACK)
    screen.blit(bgImg, (0, 0))

    manager.update(time_delta)
    manager.draw_ui(screen)
    sas.rover.obstacles.draw()
    sas.rover.waypoints.draw()
    sas._mape_loop()

    if save:
        pygame_screen_recorder.click(screen)
    # r.obstacles.draw()
    # r.waypoints.draw()
    # r.mape_loop()
    pygame.display.flip()
    fc+=1
pygame_screen_recorder.save()
pygame.quit()
