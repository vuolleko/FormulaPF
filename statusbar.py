from __future__ import division
import pygame

import statusbox
import constants

class Status_bar(pygame.sprite.Sprite):
    def __init__(self, car_list):
        super(Status_bar, self).__init__()
        self.car_list = car_list

        self.image = pygame.Surface((constants.WIDTH_STATUS, constants.HEIGHT_STATUS)).convert()
        self.image.fill(constants.WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = constants.WIDTH_TRACK

        self._draw_title()
        self.box_yoffset = 100
        self.height_box = (constants.HEIGHT_STATUS - self.box_yoffset) / len(self.car_list)
        self.frame_counter = 0

    def _draw_title(self):
        title_font = pygame.font.Font(None, 50)
        title_rend = title_font.render("FormulaAI", 1, (10, 20, 30))
        title_pos = title_rend.get_rect()
        title_pos.x += 10
        self.image.blit(title_rend, title_pos)

    def update(self):
        self.frame_counter += 1
        self._draw_timer()
        for ii, car in enumerate(self.car_list):
            self._draw_status_box(ii, car)

    def _draw_timer(self):
        minutes = self.frame_counter // (60 * constants.FRAME_RATE)
        seconds = self.frame_counter / constants.FRAME_RATE - minutes * 60

        timer_box = pygame.Surface((constants.WIDTH_STATUS, self.box_yoffset/2))
        timer_box.fill(constants.WHITE)
        box_x = 20
        box_y = 50
        self.image.blit(timer_box, (box_x, box_y))

        timer_font = pygame.font.Font(None, 50)
        timer_text = "{0:02.0f}:{1:05.2f}".format(minutes, seconds)
        timer_rend = timer_font.render(timer_text, 1, constants.COLOR_TEXT)
        self.image.blit(timer_rend, (box_x, box_y))

    def _draw_status_box(self, ii, car):
        status_box = pygame.Surface((constants.WIDTH_STATUS, self.height_box))
        status_box.fill(car.color)
        box_x = 0
        box_y = self.box_yoffset+ii*self.height_box
        self.image.blit(status_box, (box_x, box_y))

        name_font = pygame.font.Font(None, 40)
        name_font.set_bold(True)
        name_rend = name_font.render(car.name, 1, constants.COLOR_TEXT)
        self.image.blit(name_rend, (box_x, box_y))
        box_y += 50

        status_font = pygame.font.Font(None, 25)
        status_text_list = []
        status_text_list.append("Crashes: {}".format(car.crashes))
        status_text_list.append("Laps: {}".format(car.laps))
        status_text_list.append("Lap Distance: {:4.1f}".format(car.distance_try))
        status_text_list.append("Total Distance: {:4.1f}".format(car.distance_total))
        for status_text in status_text_list:
            status_rend = status_font.render(status_text, 1, constants.COLOR_TEXT)
            self.image.blit(status_rend, (box_x, box_y))
            box_y += 20
