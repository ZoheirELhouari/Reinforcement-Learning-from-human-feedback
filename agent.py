import tensorflow as tf
from network import PolicyGradientNetwork, Rewarder
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import random
import numpy as np



class ReinforceAgent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=5,layer1_size=256, layer2_size=256):
        self.gamma = gamma
        self.learning_rate = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.learning_rate))
        self.rewarder = Rewarder(layer1_size)
        self.reward_loss = 0

    def take_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample() 
        return action.numpy()[0][0]

    def save_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def train(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        
        
        print("select preference:\n 1:first\n 2:second\n 0:same")
        reward1 = []
        reward2 = []
        pref=input()
        if pref == "1" :
            dist=[1,0]
        if pref == "2" :
            dist=[0,1]
        if pref == "0" :
            dist=[1,1]
        with tf.GradientTape()as tape:
            # optimizing the rewarder for the next state 
            for j in range(len(self.state_memory) // 2):
              reward1.append(self.rewarder(self.state_memory[j]))
            for k in range(len(self.state_memory) // 2,len(self.state_memory)):
              reward2.append(self.rewarder(self.state_memory[k]))
            p1 = tf.exp(tf.math.reduce_sum(reward1))/tf.exp(tf.math.reduce_sum(reward1))+tf.exp(tf.math.reduce_sum(reward2))
            p2 = tf.exp(tf.math.reduce_sum(reward2))/tf.exp(tf.math.reduce_sum(reward1))+tf.exp(tf.math.reduce_sum(reward2))
            loss =- tf.math.log(p1)*dist[0]+tf.math.log(p2)*dist[1]
        grads = tape.gradient(loss,self.rewarder.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads,self.rewarder.trainable_variables))
        print("this is the loss",loss)
        self.reward_loss = loss
        self.state_memory = []
        self.action_memory = []

        return self.reward_loss

    def update_preference(self,states,actions,rewarder):
        pass # TODO

