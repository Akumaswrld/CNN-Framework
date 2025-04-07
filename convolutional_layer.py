import cupy as cp 
from torch.nn.functional import conv2d
import torch
import numpy as np 
from adam_optimiser import AdamOptimiser

class ConvolutionalLayer:
    def __init__(self, input_tensor_no, output_tensor_no, kernel_size, learning_rate) -> None:
        self.__kernels = cp.random.randn(output_tensor_no, input_tensor_no, kernel_size, kernel_size).astype(cp.float32) * cp.sqrt(2.0 / input_tensor_no)
        self.__biases = cp.zeros(output_tensor_no).astype(cp.float32)
        self.__derivatives = {'dL_dK' : cp.zeros_like(self.__kernels),
                              'dL_dB' : cp.zeros_like(self.__biases)}
        
        self.__optimiser = AdamOptimiser(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, w_dims=self.__kernels.shape, b_dims=self.__biases.shape)


    def forward(self, inputs) -> cp.ndarray:
        self.__inputs = inputs

        # convert inputs, weights and biases into torch tensors 
        inputs = torch.from_numpy(np.asarray(self.__inputs.get())).to(dtype=torch.float32)
        weights = torch.from_numpy(np.asarray(self.__kernels.get())).to(dtype=torch.float32)
        biases =  torch.from_numpy(np.asarray(self.__biases.get())).to(dtype=torch.float32)

        
        self.__pre_activations = cp.asarray(conv2d(inputs, weights, bias = biases, stride = 1, padding = 1))

        self.__activations = self.relu(self.__pre_activations)

        return self.__activations
        

    def backward(self, delta, n : int) -> cp.ndarray:

        # if delta isnt 4 dimensional, that means that the error has come from a dense layer meaning that we need to reshape it correctly
        if len(delta.shape) != 4:
            delta = cp.reshape(delta, self.__activations.shape)

        # scale delta by the derivative of the activation function to make the gradients more accurate by considering the activation function of the layer
        delta *= self.relu_derivative(self.__pre_activations)


        # flip inputs and delta so that 
        delta_torch = torch.from_numpy(np.asarray(delta.get())).to(dtype=torch.float32)
        inputs_torch = (torch.from_numpy(np.asarray(self.__inputs.get())).to(dtype=torch.float32)).permute(1, 0, 2, 3)
        delta_flipped = delta_torch.permute(1,0,2,3)


        # calculate derivatives of loss function wrt kernels (dL_dK) and biases (dL_dB)
        self.__derivatives['dL_dK'] = (1/n) * cp.asarray(conv2d(delta_flipped, inputs_torch, stride=1, padding=1))
        self.__derivatives['dL_dB'] = (1/n) * cp.sum(delta, axis=(0,2,3))


        # calculate the propagated error (delta new)
        flipped_weights = (torch.from_numpy(np.array(self.__kernels.get())).to(dtype=torch.float32)).permute(1,0,2,3)
        delta_new = cp.asarray(conv2d(delta_torch, flipped_weights, padding=1, stride=1))

        return delta_new
        

    def update_params(self) -> None:
        # use the instance of the adam optimiser to get the scale required to update the parameters
        derivatives = (self.__derivatives['dL_dK'], self.__derivatives['dL_dB'])
        update = self.__optimiser.get_update(derivatives=derivatives)

        # apply the scale to the parameters to update them (hopefully making them more accurate depending on how the network performs)
        self.__kernels -= update[0]
        self.__biases -= update[1]

        # zero the gradients so that they are not accumulated after every iteration which could affect the learning process
        self.__derivatives = {'dL_dK' : cp.zeros_like(self.__kernels),
                              'dL_dB' : cp.zeros_like(self.__biases)}
    
    @staticmethod
    def relu(pre_activations) -> cp.ndarray:
        return cp.maximum(0, pre_activations)

    @staticmethod
    def relu_derivative(pre_activations) -> cp.ndarray:
        return cp.where(pre_activations > 0, 1, 0)
    
    def get_weights(self):
        return self.__kernels
    
    def get_biases(self):
        return self.__biases