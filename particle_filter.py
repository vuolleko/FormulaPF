# Thanks to Andreas Svensson for inspiration!

import numpy as np
import pygame

import constants


class PFilter:
    """
    Implements a particle filter for positioning a car on track.
    """
    def __init__(self, track, car, n_particles=100, sd=10.):
        self.track = track
        self.car = car
        self.n_particles = n_particles
        self.sd = sd

        self.xx = np.zeros(n_particles, dtype=np.int)
        self.yy = np.zeros(n_particles, dtype=np.int)
        self.distribute_init()
        self.weights = np.zeros(n_particles) - np.log(self.n_particles)  # log

    def distribute_init(self):
        """
        Position particles randomly on track.
        """
        while True:
            off_grid = self.track.off_track(self.xx, self.yy)
            if not np.any(off_grid):
                break
            self.xx[off_grid] = np.random.randint(0, constants.WIDTH_TRACK, sum(off_grid))
            self.yy[off_grid] = np.random.randint(0, constants.HEIGHT_TRACK, sum(off_grid))

    def view(self, ii):
        """
        View from point, assuming car's orientation.
        """
        cos_angles = np.cos(self.car.direction + self.car.driver.view_angles)
        view_x = (self.xx[ii] + np.outer(cos_angles, self.car.driver.view_distances)
                      ).astype(int)

        sin_angles = np.sin(self.car.direction + self.car.driver.view_angles)
        view_y = (self.yy[ii] - np.outer(sin_angles, self.car.driver.view_distances)
                      ).astype(int)

        # limit coordinates within track area (only for checking if off track)
        x_matrix0 = np.where((view_x < 0) |
                             (view_x >= constants.WIDTH_TRACK),
                             0, view_x)
        y_matrix0 = np.where((view_y < 0) |
                             (view_y >= constants.HEIGHT_TRACK),
                             0, view_y)

        return self.track.off_track(x_matrix0, y_matrix0)

    def update(self, frame_counter):
        """
        Update particle weights based on a measure of similarity of view.
        """
        if frame_counter % constants.PARTICLE_UPDATE_INTERVAL == 0:
            for ii in range(self.n_particles):
                dissimilarity = np.sum(self.car.driver.view_field != self.view(ii))
                self.weights[ii] = -(dissimilarity**2 + 1) / 500.

            # print self.weights
            weights = np.exp(self.weights)
            weights /= np.sum(weights)

            # resample particles
            indices = np.random.choice(np.arange(self.n_particles),
                                       size=self.n_particles, replace=True,
                                       p=weights)

            xx = (self.xx[indices]
                  + self.car.speed * np.cos(self.car.direction) * constants.PARTICLE_UPDATE_INTERVAL
                  + np.random.randn(self.n_particles) * self.sd)
            yy = (self.yy[indices]
                  - self.car.speed * np.sin(self.car.direction) * constants.PARTICLE_UPDATE_INTERVAL
                  + np.random.randn(self.n_particles) * self.sd)
            xx = np.where(xx < 0, 0, xx)
            xx = np.where(xx >= constants.WIDTH_TRACK, constants.WIDTH_TRACK-1, xx)
            yy = np.where(yy < 0, 0, yy)
            yy = np.where(yy >= constants.HEIGHT_TRACK, constants.HEIGHT_TRACK-1, yy)
            self.xx = np.int0(xx)
            self.yy = np.int0(yy)


    def draw(self, screen):
        """
        Draw the particles.
        """
        for xx, yy in zip(self.xx, self.yy):
            pygame.draw.circle(screen, constants.GREEN, (xx, yy), 3)
