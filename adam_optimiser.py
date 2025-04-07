import cupy as cp 

class AdamOptimiser:
    def __init__(self, learning_rate, beta_1, beta_2, epsilon, w_dims, b_dims):
        # initialise hyperparameters 
        self.__learning_rate = learning_rate
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2 
        self.__epsilon = epsilon

        # initialise running mean and variance
        self.__mean = [cp.zeros(w_dims), cp.zeros(b_dims)]
        self.__variance = [cp.zeros(w_dims), cp.zeros(b_dims)]

        self.__iteration_no = 0

    def get_update(self, derivatives):
        # idx 0 refers to weights and idx 1 refers to biases 
        updates = []
        self.__iteration_no += 1

        for idx in range(2):
            self.__mean[idx] = self.__beta_1 * self.__mean[idx] + (1 - self.__beta_1) * derivatives[idx]
            self.__variance[idx] = self.__beta_2 * self.__variance[idx] + (1 - self.__beta_2) * (derivatives[idx] ** 2)

            mean_corrected = self.__mean[idx] / (1 - (self.__beta_1 ** self.__iteration_no))
            variance_corrected = self.__variance[idx] / (1 - (self.__beta_2 ** self.__iteration_no))

            # take away this result from the corresponding parameter and that is the updated parameter
            update = (self.__learning_rate * mean_corrected) / (cp.sqrt(variance_corrected) + self.__epsilon)
            updates.append(update)
            
        return updates