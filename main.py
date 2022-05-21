import os
import sys

import matplotlib
import pygame
import numpy as np
from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
pygame.init()


class Button:
    def __init__(self, surface, text, x, y, h, w, font, size, sign=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.size = size
        self.surface = surface
        self.sign = sign

    def set_text(self, text):
        self.text = text

    def draw_button(self):
        # If it is sign we just want to have one color
        if self.sign == True:
            color = (180, 210, 180)
        # If it is button
        else:
            # if mouse is hovered on a button, different color and different if it is not
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                color = (100, 210, 100)
            else:
                color = (180, 210, 180)

        pygame.draw.rect(self.surface, color, self.rect)
        points = [self.rect.topleft, self.rect.topright, self.rect.bottomright, self.rect.bottomleft]
        pygame.draw.lines(self.surface, (0, 0, 0), True, points, 3)

        font = pygame.font.Font(self.font, self.size)
        text = font.render(self.text, True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = self.rect.center
        self.surface.blit(text, textRect)


def predict_digit(image, model):
    # Next 4 steps are used to resize our image to fit the model's requirements [1, 28, 28]
    image = image[0:784, 0:784]
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (28, 28), interpolation=cv.INTER_AREA)
    image = np.reshape(image, (1, 28, 28))

    # Prediction
    prob = model.predict(image)
    prob.round(2)
    prob = np.argmax(prob, axis=1)
    return prob[0]


def main():
    # Get model and font
    model = keras.models.load_model('model')
    font = os.path.join('Fonts', 'Louis George Cafe Bold.ttf')

    size = 784 # later we need to resize image to 28, so size of our surface must be multiplication of 28
    surface = pygame.display.set_mode((size + 250, size))  # initializing surface
    pygame.display.set_caption("Digit Recognizer")  # Set title
    surface.fill((0, 0, 0))

    menu = pygame.Rect(785, 0, 250, 784) # Space for buttons

    # init buttons
    clear_button = Button(surface, 'Clear board', 795, 20, 55, 229, font, 35)
    predict_button = Button(surface, 'Predict', 795, 95, 55, 229, font, 35)
    predicted_digit_text = Button(surface, 'Predicted number', 795, 170, 55, 229, font, 27, True)
    predicted_digit = Button(surface, '', 795, 225, 200, 229, font, 80, True)
    exit_button = Button(surface, 'Exit', 795, 709, 55, 229, font, 35)
    draw = False

    # width of drawn lines
    width = 8

    while True:
        # draw everything on surface
        pygame.draw.rect(surface, (0, 255, 0), menu)
        clear_button.draw_button()
        predict_button.draw_button()
        predicted_digit.draw_button()
        predicted_digit_text.draw_button()
        exit_button.draw_button()
        pygame.display.update()

        # event handler
        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # we want to draw only if we are in space that is intended for this and if mouse button is pressed
            if (0, 0) <= pos <= (784 - width, 784 - width):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    draw = True
            else:
                # if button clicked
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # clear board
                    if clear_button.rect.collidepoint(pos):
                        surface.fill((0, 0, 0))
                    # predict drawn digit
                    if predict_button.rect.collidepoint(pos):
                        pygame.image.save_extended(surface, "number.png")
                        pygame.display.update()
                        image = cv.imread("number.png")
                        digit = predict_digit(image, model)
                        predicted_digit.set_text(str(digit))
                    # exit
                    if exit_button.rect.collidepoint(pos):
                        pygame.quit()
                        sys.exit()
                draw = False
            if event.type == pygame.MOUSEBUTTONUP:
                draw = False

        if draw:
            # we are drawing our digit by continuously drawing circles
            pygame.draw.circle(surface, (255, 255, 255), pygame.mouse.get_pos(), width)


if __name__ == '__main__':
    main()
    # test()
