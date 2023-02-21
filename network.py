import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 

class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, first_hidden_units=256, second_hidden_units=256):
        super(PolicyGradientNetwork, self).__init__()

        self.first_dims = first_hidden_units
        self.second_dims = second_hidden_units
        self.n_actions = n_actions

        self.fisrt_layer = Dense(self.first_dims, activation='relu')
        self.second_layer = Dense(self.second_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fisrt_layer(state)
        value = self.second_layer(value)
        pi = self.pi(value)

        return pi

class Rewarder(keras.Model):
    def __init__(self,num_hidden_units= 256):
        super().__init__()
        self.shared_1 = Dense(num_hidden_units,activation='relu')
        self.reward  = Dense(1,activation='linear')

    def call(self,input_obs):
        x = self.shared_1(input_obs)
        return self.reward(x)