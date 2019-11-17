import numpy as np
from copy import deepcopy

class KalmanFilter(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def transit(self, A, R):
        self.mu = np.dot(A, self.mu)
        rr =np.dot(A, self.sigma)
        self.sigma = np.dot(rr, A.T) + R

    def process_observation(self, observation, C, Q):
        assert observation.shape[0] == C.shape[0], '{}, {}'.format(observation.shape[0], C.shape[0])
        assert C.shape[0] == Q.shape[0]
        H = np.linalg.inv(np.dot(np.dot(C, self.sigma), C.T) + Q)
        K = np.dot(np.dot(self.sigma, C.T), H)
        self.mu += np.dot(K, observation - np.dot(C, self.mu))
        self.sigma = np.dot(np.eye(K.shape[0]) - np.dot(K, C), self.sigma)
        self.sigma[np.abs(self.sigma) < 1e-16] = 0


class Car(object):
    POS_X_INDEX = 0
    POS_Y_INDEX = 1
    POS_Z_INDEX = 2
    VEL_X_INDEX = 3
    VEL_Y_INDEX = 4
    VEL_Z_INDEX = 5

    def __init__(self, initial_position=None, initial_velocity=None, time=0):
        if initial_position is None:
            initial_position = np.zeros(3, dtype=np.float64)
        else:
            initial_position = np.array(initial_position, dtype=np.float64)
            assert initial_position.shape == (3,)
        if initial_velocity is None:
            initial_velocity = np.zeros(3, dtype=np.float64)
        else:
            initial_velocity = np.array(initial_velocity, dtype=np.float64)
            assert initial_velocity.shape == (3,)

        self._state = np.hstack((initial_position, initial_velocity))
        self._position_x, self._position_y, self._position_z = initial_position[:]
        self._velocity_x, self._velocity_y, self._velocity_z = initial_velocity[:]

        self._time = time
        sigma = self.get_noise_covariance()
        self.kalman = KalmanFilter(mu=self._state, sigma=sigma)

    def get_transition_matrix(self, dt):

        transition_matrix = np.eye(6,6)
        transition_matrix[self.POS_X_INDEX, self.VEL_X_INDEX] = dt  # pos_x <- x_vel
        transition_matrix[self.POS_Y_INDEX, self.VEL_Y_INDEX] = dt  # pos_y <- y_vel
        transition_matrix[self.POS_Z_INDEX, self.VEL_Z_INDEX] = dt  # pos_y <- y_vel
        return transition_matrix

    def move(self, dt):
        A = self.get_transition_matrix(dt)
        self._state = np.dot(A, self._state)
        self._time = self._time + dt

    def get_noise_covariance(dt=None):
        return np.diag([2.60078734e-04, 5.74213284e-04, 9.07507921e-01, \
                        6.91341016e-06, 1.85054632e-05, 5.85166217e-02])
    def get_observation_matrix(self):
        '''!!!
        '''
        observation_matrix = np.zeros((6, 6), dtype=np.float64)
        observation_matrix[0, self.POS_X_INDEX] = 1
        observation_matrix[1, self.POS_Y_INDEX] = 1
        observation_matrix[2, self.POS_Z_INDEX] = 1
        return observation_matrix



class TimeSmoother:
    def __init__(self, pos, vel, ts, horizon=10*1000):
        self.car_real = Car(pos, vel, ts)
        self.car_kalman = Car(pos, vel, ts)
        self.horizon = horizon

    def __car_update(self, car, state, ts):
        C = car.get_observation_matrix()[:6]
        Q = car.get_noise_covariance()[:6, :6]
        car.kalman.process_observation(state, C, Q)
        return car

    def kalman_predict(self, ts):
        pred_car = deepcopy(self.car_kalman)
        pred_car.move(ts - pred_car._time)
        return pred_car._state

    def kalman_update(self, ts, state_vector, predict_vector):
        self.car_real = self.__car_update(self.car_real, state_vector, ts)
        self.car_kalman = deepcopy(self.car_real)
        self.car_kalman = self.__car_update(self.car_real, predict_vector, ts + self.horizon)
