import random
import os
import pygame
import gsn.gsn as gsn
import nn.ImgNet as nn
import torch
import numpy as np
import math
from kb import KnowledgeBase
os.environ["SDL_VIDEODRIVER"] = "directfb"

WIDTH = 500
HEIGHT = 500
DIAG = (WIDTH**2 + HEIGHT**2) ** 0.5
FPS = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)


def rotateVector(v, theta):
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )  # https://www.atqed.com/numpy-rotate-vector
    return rot.dot(v)


class Obstacle:
    def __init__(self, x, y, label, image, screen):
        self.x = x
        self.y = y
        self.label = label
        self.image = image
        self.screen = screen

    def draw(self):
        # pygame_img = pygame.surfarray.make_surface(
        #     self.image[0].permute(*torch.arange(self.image[0].ndim - 1, -1, -1)).numpy()
        # )
        # self.screen.blit(pygame_img, (self.x, self.y))
        pygame.draw.circle(self.screen, BLUE, (self.x, self.y), 20)


class Obstacles:
    def __init__(self, screen):
        self.data = list(iter(nn._get_test_data()))
        self.obstacles = []
        for img, label in self.data:
            self.obstacles.append(
                Obstacle(
                    random.randint(0, WIDTH - 20),
                    random.randint(0, HEIGHT - 20),
                    label,
                    img,
                    screen,
                )
            )
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

class Rover:
    def __init__(self, waypoints, obstacles, model, screen):
        # general params
        self.screen = screen
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
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
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.center = (self.x, self.y)
        self.surface.fill(YELLOW)
        # battery params
        self.battery_level = 1000
        self.battery_delta = 10
        # mape-K params
        self.MODE = "drive"
        # self.knowledge_base = KnowledgeBase("gsn/gsn_2.xml", "model/ImgNet.pth")
        self.waypoints = waypoints
        self.obstacles = obstacles
        # self.gsnTree = self.knowledge_base.gsn
        self.shouldUpdateDirection = True
        self.model = model
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
        self.x += self.speedx * self.delta
        self.y += self.speedy * self.delta
        self.rect.center = (self.x, self.y)

    def move_manual(self, dx, dy):
        self.x += dx
        self.y += dy
        self.rect.center = (self.x, self.y)

    def rotate(self):
        ang = np.arccos(np.dot([self.speedx, self.speedy], [0, 1])) * (180 / np.pi)
        tmp = self.surface.copy()
        return pygame.transform.rotate(tmp, ang)

    def draw_view_range(self):
        # get top left and top right of the triangle
        adj = np.array([self.speedx, self.speedy]) * self.view_range
        theta = np.pi / 6
        adj1 = rotateVector(adj, theta)
        adj2 = rotateVector(adj, -theta)
        cx, cy = self.rect.center
        pygame.draw.line(self.screen, WHITE, (cx, cy), (cx + adj2[0], cy + adj2[1]), 5)
        pygame.draw.line(self.screen, WHITE, (cx, cy), (cx + adj1[0], cy + adj1[1]), 5)
        pygame.draw.line(self.screen, WHITE, (cx + adj1[0], cy + adj1[1]), (cx + adj2[0], cy + adj2[1]), 5)
        

    def draw(self):
        # pygame.draw.rect(self.screen, GREEN, self.rect)
        rot_img = self.rotate()
        temp_rect = rot_img.get_rect(center=(self.rect.center))
        self.screen.blit(rot_img, temp_rect)
        self.draw_view_range()
        if self.MODE == "charge":
            battery_surface = pygame.image.load("assets/solar.png")
            battery_surface = pygame.transform.scale(
                battery_surface, (self.width, self.height)
            )
            battery_rect = battery_surface.get_rect(center=(self.rect.center))
            self.screen.blit(battery_surface, battery_rect)

    def print_debug_logs(self):
        os.system("clear")
        obstacle = self.obstacles.get_current_obstacle()
        dist_to_obstacle = np.linalg.norm(
            [obstacle.x - self.rect.center[0], obstacle.y - self.rect.center[1]]
        )

        str_c = (
            f"current waypoint: {self.waypoints.current_waypoint_id}\n"
            f"distanct to waypoint: {self.get_distance_to_current_waypoint()}\n"
            f"battery level: {self.battery_level}\n"
            f"battery utility violated: {self.isUtilityViolatedBattery}\n"
            f"cv utility violated: {self.isUtilityViolatedCV}\n"
            f"MODE: {self.MODE}\n"
            f"currentl velocity: {self.speedx, self.speedy}\n"
            f"CV Degraded: {self.CvDegraded}\n"
            f"pred_debug logs: {self.pred_debug_logs}\n"
            f"current obstacle dist: {dist_to_obstacle}\n"
            f"obstacles detected: {self.obstacles.current_obstacle_index}\n"
        )
        print(str_c)
        return str_c


    def update_direction_to_waypoint(self):
        waypoint = self.waypoints.get_current_waypoint()
        xdir = waypoint.x - self.x
        ydir = waypoint.y - self.y
        mag = math.sqrt(xdir**2 + ydir**2)
        delta = 10
        self.speedx = xdir / mag
        self.speedy = ydir / mag

    def reset_battery_level(self):
        self.battery_level = 1000

    def update_battery_level(self):
        self.battery_level -= self.battery_delta

    def get_speed(self):
        if self.speedx == 0 and self.speedy == 0:
            return 0
        return math.sqrt(self.speedx**2 + self.speedy**2) * self.delta

    def make_prediction(self, obstacle):
        outputs = self.model(obstacle.image)
        _, predicted = torch.max(outputs, 1)
        # calculate accuracy predicted vs labels
        correct = len(
            list(filter(lambda x: x[0] == x[1], zip(predicted, obstacle.label)))
        )
        accuracy = correct / len(predicted)
        self.pred_debug_logs = f"Accuracy: {accuracy}"
        if accuracy > self.cv_threshold:
            return True, accuracy
        return False, accuracy

    def check_obstacle_in_radar(self):
        obstacle = self.obstacles.get_current_obstacle()
        vec_rover_to_obstacle = np.array(
            [obstacle.x - self.rect.center[0], obstacle.y - self.rect.center[1]]
        )
        vec_rover_to_obstacle = vec_rover_to_obstacle / np.linalg.norm(
            vec_rover_to_obstacle
        )
        angle_from_speed = np.cos(
            np.dot(vec_rover_to_obstacle, np.array([self.speedx, self.speedy]))
        )
        dist_to_obstacle = np.linalg.norm(
            [obstacle.x - self.rect.center[0], obstacle.y - self.rect.center[1]]
        )
        if (
            angle_from_speed <= np.pi / 2
            and abs(dist_to_obstacle - self.view_range) <= 20
        ):
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
        return ((self.x - waypoint.x) ** 2 + (self.y - waypoint.y) ** 2) ** 0.5

    def check_rover_reaches_waypoint(self):
        waypoint = self.waypoints.get_current_waypoint()
        dist = self.get_distance_to_current_waypoint()
        if dist < waypoint.radius:
            self.waypoints.update_current_waypoint()
            self.shouldUpdateDirection = True
            if self.MODE == "manual":
                self.MODE = "drive"

    def get_battery_data(self):
        batteryLevel = self.battery_level
        batteryNeeded = self.get_distance_to_current_waypoint()
        return batteryLevel, batteryNeeded

    def get_cv_data(self):
        return self.CvDegraded


class Waypoint:
    def __init__(self, x, y, screen, radius=5):
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


class Waypoints:
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
