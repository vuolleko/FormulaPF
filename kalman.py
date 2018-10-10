import numpy as np
import pygame
import scipy.stats as ss

import constants


class Kalman:
    """Kalman filter for positioning the car.

    The predicted state x_hat is [pos_x, pos_y].
    Control input u is [velocity_x, velocity_y].
    Measurement z is [pos_x, pos_y] with uncertainty.
    """
    def __init__(self, track, car, cov_process=np.diag((1., 1.)), cov_obs=np.diag((10000, 10000))):
        self.track = track
        self.car = car
        self.Q = cov_process
        self.R = cov_obs
        dt = 1.

        # initial position known with certainty
        self.x_hat = np.array([car.pos_x, car.pos_y])[:, None]  # predicted state estimate
        self.P = np.diag((0.1, 0.1))  # predicted error cov

        # state-transition, control-input and observation models
        self.F = np.array([[1., 0], [0, 1]])
        self.B = np.array([[dt, 0], [0, dt]])
        self.H = np.array([[1., 0], [0, 1.]])  # position observed directly

    def control(self):
        """
        Return control signal.
        """
        u = np.array([[self.car.speed * np.cos(self.car.direction), -self.car.speed * np.sin(self.car.direction)]]).T
        w = ss.multivariate_normal.rvs(cov=self.Q)[:, None]
        return u + w

    def observe(self):
        """
        Return observed position of the car.
        """
        x = np.array([self.car.pos_x, self.car.pos_y])[:, None]
        v = ss.multivariate_normal.rvs(cov=self.R)[:, None]
        z = self.H.dot(x) + v
        return z

    def update(self, *args):
        """
        Predict and update.
        """
        u = self.control()

        # prediction step
        self.x_hat = self.F.dot(self.x_hat) + self.B.dot(u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

        z = self.observe()

        # update step
        y = z - self.H.dot(self.x_hat)  # y = z - Hx
        S = self.R + self.H.dot(self.P).dot(self.H.T)  # S = R + HPH.T
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))  # K = P H.T inv(S)
        self.x_hat = self.x_hat + K.dot(y)  # x = x + Ky
        self.P = (np.eye(*self.P.shape) - K.dot(self.H)).dot(self.P)  # P = (I -KH)P

    def draw(self, screen):
        """
        Draw the max a posteriori and uncertainty.
        """
        pygame.draw.circle(screen, constants.BLACK, (int(self.x_hat[0, 0]), int(self.x_hat[1, 0])), 5)
        if self.P[0, 0] > 1:
            pygame.draw.circle(screen, constants.BLACK, (int(self.x_hat[0, 0]), int(self.x_hat[1, 0])), int(np.sqrt(5.991*self.P[0, 0])), 1)
