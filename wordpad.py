import pygame
from pygame.locals import *
import sys
from PIL import Image
import cv2
import numpy as np
from resizeimage import resizeimage
import tensorflow as tf

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
FPS = 32
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
RADIUS = 10
# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Loading our model
model = tf.keras.models.load_model("digits_model")


def digPredict(gameDisplay):
    '''
    Digits prediction
    '''
    data = pygame.image.tostring(gameDisplay, 'RGBA')
    img = Image.frombytes('RGBA', (SCREEN_WIDTH, SCREEN_HEIGHT), data)
    img = resizeimage.resize_cover(img, [28, 28])
    imgobj = np.asarray(img)
    imgobj = cv2.cvtColor(imgobj, cv2.COLOR_RGB2GRAY)
    (_, imgobj) = cv2.threshold(imgobj, 128,
                                255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Predicting
    imgobj = imgobj/255
    b = model.predict(np.reshape(
        imgobj, [1, imgobj.shape[0], imgobj.shape[1], 1]))
    print('*'*15)
    print(f'b : {np.argmax(b)}')
    print('b', b, 'b[0]', b[0][np.argmax(b)])
    ans = (np.argmax(b) if b[0][np.argmax(b)] > 0.5 else '?')
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN or event.type == MOUSEBUTTONDOWN:
                SCREEN.fill(BLACK)
                pygame.display.update()
                return
        SCREEN.fill(BLACK)
        text_surface = font.render(f"Predicted Value: {ans}", True, RED)
        SCREEN.blit(text_surface, (10, 100))
        pygame.display.update()


if __name__ == '__main__':
    pygame.init()
    font = pygame.font.Font('freesansbold.ttf', 30)
    FPS_CLOCK = pygame.time.Clock()
    pygame.display.set_caption('WordPad')
    SCREEN.fill(BLACK)
    pygame.display.update()
    FPS_CLOCK.tick(FPS)
    gameExit = False
    while not gameExit:
        for event in pygame.event.get():
            if event.type == QUIT:
                gameExit = True
            if event.type == KEYDOWN and event.key == K_RETURN:
                digPredict(SCREEN)
            if event.type == KEYDOWN and event.key == K_c:
                SCREEN.fill(BLACK)
                pygame.display.update()

        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(SCREEN, WHITE, spot, RADIUS)
            pygame.display.flip()

    pygame.quit()
    sys.exit()
