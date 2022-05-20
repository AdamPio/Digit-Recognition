import sys

import matplotlib
import pygame
import numpy as np
from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def test():
    model = keras.models.load_model('model')
    image = cv.imread('test.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (28, 28), interpolation=cv.INTER_AREA)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.show()
    image = np.reshape(image, (1, 28, 28))
    prob = model.predict(image)
    prob.round(2)
    prob = np.argmax(prob, axis=1)
    print(prob)


def main():
    size = 784
    surface = pygame.display.set_mode((size, size))  # initializing surface
    pygame.display.set_caption("Digit Recognizer")  # Set title
    surface.fill((0, 0, 0))
    draw = False

    while True:
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(surface, "test.png")
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                draw = True
            if event.type == pygame.MOUSEBUTTONUP:
                draw = False

        if draw:
            # surface.fill((255, 255, 255), (pygame.mouse.get_pos(), (1, 1)))
            pygame.draw.circle(surface, (255, 255, 255), pygame.mouse.get_pos(), 8)


if __name__ == '__main__':
    # main()
    test()
