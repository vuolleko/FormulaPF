import numpy as np
import pygame

import constants
from particle_filter import PFilter


class BO(PFilter):
    """
    Implements Bayesian optimization for positioning the car on track.
    """
    def __init__(self, track, car, n_initial=30, theta=100.):
        super(BO, self).__init__(track, car, n_particles=n_initial)
        self.theta_inv = 1. / theta

        # evaluate similarities for the initial sample
        self.similarities = np.zeros(0)
        for ii in range(n_initial):
            self.add_similarity(ii)
        self.n_particles = n_initial
        self.find_proposals()

    def find_proposals(self):
        """
        Find proposal coordinates that are on track.
        """
        xs = np.arange(0, constants.WIDTH_TRACK, 5)
        ys = np.arange(0, constants.HEIGHT_TRACK, 5)
        xs, ys = np.meshgrid(xs, ys)
        on_track = np.invert( self.track.off_track(xs, ys) )
        self.x_prop = xs[on_track].ravel()
        self.y_prop = ys[on_track].ravel()
        self.mus = np.zeros(self.x_prop.shape)  # for posterior predictive means
        self.sigmas = np.zeros(self.x_prop.shape)  # for posterior predictive variances

    def add_similarity(self, ii):
        """
        Similarity of view.
        """
        dissim = np.sum(self.car.driver.view_field != self.view(ii))
        self.similarities = np.append( self.similarities, [np.exp(-(dissim + 1) / 10.)] )

    def acquisition(self):
        """
        Acquisition function for the minimum expected dissimilarity.
        """
        eta = 2. * np.log( self.n_particles**2 * np.pi**2. / 0.3)
        acquis = self.mus + np.sqrt(eta * self.sigmas)
        return np.argmin(acquis)

    def update(self, frame_counter):
        """
        Perform Bayesian optimization.
        Dissimilarity is assumed constant in time?
        """
        if frame_counter % constants.BAYES_UPDATE_INTERVAL == 0:
            K_matrix = kernel( *( np.meshgrid(self.xx, self.xx)
                                  + np.meshgrid(self.yy, self.yy)
                                  + [self.theta_inv] ) )
            try:
                K_inv = np.linalg.inv(K_matrix)
            except:  # a nasty hack to avoid crashing due to singular matrix
                K_inv = np.linalg.inv(K_matrix + np.random.randn(self.n_particles, self.n_particles))

            for ii in range(len(self.x_prop)):
                small_k = kernel(self.x_prop[ii], self.xx, self.y_prop[ii], self.yy, self.theta_inv)
                kTK_inv = small_k.dot(K_inv)
                self.mus[ii] = kTK_inv.dot(self.similarities)
                self.sigmas[ii] = 1. - kTK_inv.dot(small_k)

            self.n_particles += 1

            ii_best = self.acquisition()
            self.xx = np.append(self.xx, [self.x_prop[ii_best] + int(np.random.randn()*50)])
            self.yy = np.append(self.yy, [self.y_prop[ii_best] + int(np.random.randn()*50)])
            self.add_similarity(self.n_particles - 1)

            self.move_particles(deltaTime=constants.BAYES_UPDATE_INTERVAL)

    def draw(self, screen):
        """
        Draw the particle with the highest posterior probability.
        """
        # super(BO, self).draw(screen)
        ii_best = np.argmax(self.mus)
        pygame.draw.circle(screen, constants.BLUE, (self.x_prop[ii_best], self.y_prop[ii_best]), 5)


def kernel(xx1, xx2, yy1, yy2, theta_inv):
    """
    The kernel function.
    """
    return np.exp(-theta_inv * ( (xx1 - xx2)**2. + (yy1 - yy2)**2. ) )
