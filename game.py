import pygame

from car import Car
import driver
from track import Track
from statusbar import Status_bar
import constants
import particle_filter
import bayes

# init stuff
pygame.init()
clock = pygame.time.Clock()
done = False
draw_viewfield = False
learn_from_player = False

screen = pygame.display.set_mode((constants.WIDTH_SCREEN,
                                  constants.HEIGHT_SCREEN))
pygame.display.set_caption("FormulaPF")

track = Track()
start_position, start_direction = track.find_start(4)

ai_tif_car = Car("AI_TIF", constants.RED, start_position[3], start_direction,
                 driver.AI_TIF())

sprite_list = pygame.sprite.Group()
car_list = pygame.sprite.Group()
for car in [ai_tif_car]:
    car_list.add(car)
    sprite_list.add(car)

status_bar = Status_bar(car_list)
sprite_list.add(status_bar)

particles = particle_filter.PFilter(track, ai_tif_car, 500, 7)
bayes = bayes.Bayes(track, ai_tif_car)

frame_counter = 0

while not done:
    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                draw_viewfield = not draw_viewfield
            elif event.key == pygame.K_l:
                learn_from_player = not learn_from_player
            elif event.key == pygame.K_r:
                for car in car_list:
                    car.reset(frame_counter)
                ann_batch_car.driver.reset_samples()
            elif event.key == pygame.K_t:
                ann_batch_car.driver.train()
            elif event.key == pygame.K_p:
                paused = True
                while paused:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_p:
                                paused = False
                    clock.tick(constants.FRAME_RATE)  # fps

    # update game status and handle game logic
    car_list.update(track, frame_counter)
    particles.update(frame_counter)
    bayes.update(frame_counter)
    status_bar.update(frame_counter)

    # update draw buffer
    track.draw(screen)
    sprite_list.draw(screen)
    if draw_viewfield:
        for car in car_list:
            car.driver.draw_viewfield(screen)

    # draw particles
    particles.draw(screen)
    bayes.draw(screen)

    # update screen
    clock.tick(constants.FRAME_RATE)  # fps
    frame_counter += 1
    pygame.display.flip()


pygame.quit()
