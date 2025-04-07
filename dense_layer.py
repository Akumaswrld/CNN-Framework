import cupy as cp 
from adam_optimiser import AdamOptimiser

class DenseLayer:
    def __init__(self, inputs_no, neurons_no, activation_function, learning_rate) -> None:
        self.__weights = cp.random.randn(inputs_no, neurons_no).astype(cp.float32) * cp.sqrt(1.0 / inputs_no)
        self.__biases = cp.zeros((1, neurons_no)).astype(cp.float32)
        self.__derivatives = {'dL_dW' : cp.zeros_like(self.__weights),
                              'dL_dB' : cp.zeros_like(self.__biases)}
        self.__activation_function = activation_function
        self.__optimiser = AdamOptimiser(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, w_dims=self.__weights.shape, b_dims=self.__biases.shape)
        
        
    def forward(self, inputs):
        # flatten inputs into individual vectors if inputs are coming from a convolutional layer
        if inputs.shape[-1] != self.__weights.shape[0]:
            self.__inputs = cp.reshape(inputs, (inputs.shape[0], self.__weights.shape[0]))
        else:
            self.__inputs = inputs

        
        # perform linear tranformation to the inputs (dot with weights + biases)
        self.__pre_activations = cp.dot(self.__inputs, self.__weights) + self.__biases#


        # apply the corresponding activation function to the transformed inputs (pre activations)
        if self.__activation_function == 'relu':
            self.__activations = self.relu(self.__pre_activations)

        elif self.__activation_function == 'softmax':
            self.__activations = self.softmax(self.__pre_activations)

        elif not self.__activation_function:
            self.__activations = self.__pre_activations
            

        return self.__activations

    def backward(self, delta, n : int):
        # scale delta by the derivative of activation function so that delta also considers the derivative of the activation function before being used in operations with the layers weights
        if self.__activation_function == 'relu':
            delta *= self.relu_derivative(self.__pre_activations)

        # calculate derivative of the loss function with respect to weights and biases 
        self.__derivatives['dL_dW'] = (1/n) * cp.dot(self.__inputs.T, delta)
        self.__derivatives['dL_dB'] = (1/n) * cp.sum(delta, axis=0, keepdims=True)

        # calculate the new propagated error
        delta_new = cp.dot(delta, self.__weights.T)

        return delta_new

    def update_params(self) -> None:
        # combine the weights and biases derivatives into a single tuple ready to be used in optimiser class
        derivatives = (self.__derivatives['dL_dW'], self.__derivatives['dL_dB'])

        # get the scale required to update the parameters from the optimiser class
        update = self.__optimiser.get_update(derivatives=derivatives)


        # update the parameters
        self.__weights -= update[0]
        self.__biases -= update[1]

        # reset derivatives so that they do not accumulate 
        self.__derivatives = {'dL_dW' : cp.zeros_like(self.__weights),
                              'dL_dB' : cp.zeros_like(self.__biases)}

    @staticmethod
    def relu(pre_activations) -> cp.ndarray:
        return cp.maximum(0, pre_activations)

    @staticmethod
    def relu_derivative(pre_activations) -> cp.ndarray:
        return cp.where(pre_activations > 0, 1, 0)
    
    @staticmethod
    def softmax(pre_activations):
        exp = cp.exp(pre_activations - cp.max(pre_activations, axis=1, keepdims=True))
        return exp / cp.sum(exp, axis=1, keepdims=True)
    
    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases