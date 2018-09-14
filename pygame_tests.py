import random
import pygame
from pygame.locals import *
import os
import numpy as np

# initialise the game
pygame.init()
pygame.font.init()

# set the window size
width, height = 700, 500
screen = pygame.display.set_mode((width, height))

myfont = pygame.font.SysFont('Comic Sans MS', 30)

textsurface = myfont.render('Some Text', False, (200, 200, 200))



x_m = np.array([[5, 8, 2, 0, 5],
       [3, 1, 6, 8, 1],
       [2, 2, 6, 0, 3],
       [5, 7, 6, 7, 8],
       [5, 8, 1, 1, 4]])

while 1:
    # clear the screen before drawing it again
    # screen.fill(0)


    screen.blit(textsurface, (400, 400))
    # update the screen
    pygame.display.flip()
    # loop through the events
    for event in pygame.event.get():
        # check if the event is the X button
        if event.type == pygame.QUIT:
            # if it is quit the game
            pygame.quit()
            exit(0)