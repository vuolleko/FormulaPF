import pygame

from car import Car
import driver
from track import Track
from statusbar import Status_bar
import constants

# init stuff
pygame.init()
clock = pygame.time.Clock()
done = False
draw_viewfield = False
learn_from_player = False

screen = pygame.display.set_mode((constants.WIDTH_SCREEN,
                                  constants.HEIGHT_SCREEN))
pygame.display.set_caption("FormulaAI")

track = Track()
start_position, start_direction = track.find_start(3)


player_car = Car("Player", constants.BLUE, start_position[0], start_direction,
                 driver.Player())
ann_car = Car("ANN", constants.RED, start_position[1], start_direction,
              driver.AI_ANN(model_car=player_car))
ai_car = Car("AI", constants.GREEN, start_position[2], start_direction,
             driver.Driver())

sprite_list = pygame.sprite.Group()
car_list = pygame.sprite.Group()
for car in [player_car, ann_car, ai_car]:
    car_list.add(car)
    sprite_list.add(car)

status_bar = Status_bar(car_list)
sprite_list.add(status_bar)

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
                    car.reset()
            elif event.key == pygame.K_t:
                # train_ann()
                pass

    # update game status and handle game logic
    car_list.update(track)
    status_bar.update()

    # update draw buffer
    track.draw(screen)
    sprite_list.draw(screen)
    if draw_viewfield:
        for car in car_list:
            car.driver.draw_viewfield(screen)

    # update screen
    clock.tick(constants.FRAME_RATE)  # fps
    pygame.display.flip()


pygame.quit()
