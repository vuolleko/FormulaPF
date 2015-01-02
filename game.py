import pygame

import car
import track
import constants

pygame.init()

screen = pygame.display.set_mode((constants.SCREEN_WIDTH,
                                  constants.SCREEN_HEIGHT))
pygame.display.set_caption("FormulaAI")

track = track.Track()
startpos = track.find_start()
sprite_list = pygame.sprite.Group()
sprite_list.add(track)

player_car = car.Car(constants.BLUE, startpos)
car_list = pygame.sprite.Group()
car_list.add(player_car)
sprite_list.add(player_car)

done = False
clock = pygame.time.Clock()

while not done:
    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_car.accelerate()
    if keys[pygame.K_DOWN]:
        player_car.brake()
    if keys[pygame.K_LEFT]:
        player_car.turn(constants.TURN_SPEED)
    if keys[pygame.K_RIGHT]:
        player_car.turn(-constants.TURN_SPEED)

    # update game status
    car_list.update()

    # handle game logic
    for car in car_list:
        if track.off_track(car):
            car.reset()
        elif track.halfway(car):
            car.passed_halfway()
            print "ok!"
        # print track.off_track(car)

    # update draw buffer
    # screen.fill(constants.GREEN)
    sprite_list.draw(screen)

    # update screen
    clock.tick(60)  # 60 fps
    pygame.display.flip()


pygame.quit()