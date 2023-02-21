from time import sleep
from tkinter import N
import numpy as np
from agent import ReinforceAgent
from Enviroment import Environment
import pygame

if __name__ == '__main__':
      agent = ReinforceAgent(alpha=0.0005,  gamma=0.99,n_actions=5)
      env = Environment()
      n_episodes = 100
      pygame.init()
      WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
      WINDOW1          = pygame.display.get_wm_info()
      clock           = pygame.time.Clock()
      pygame.display.set_caption("Reinforcement Learning")
      cross_entropy = []
      for i in range(n_episodes):
        game_over = False
        observation = env.reset()
        score = 0
        score_increased = False
        count = 0
        for j in range(14):
        # while not game_over:
            action = agent.take_action(observation)
            observation_, reward, game_over = env.step(action)
            
            score += reward
            agent.save_transition(observation, action, reward)
            observation = observation_
            count += 1
            env.render(WINDOW,title="trajectory 1")
            if (game_over == True):
              break
        print("this is the count 1 ",count)
        # print("this is the reward",reward)
        
        sleep(0.5)
        game_over = False
        observation = env.reset()
        score_increased = False
        score = 0
        count = 0
        for j in range(14):
          # while not game_over:
            action = agent.choose_action(observation)
            print("action: ", action)
            observation_, reward, game_over = env.step(action)
            # env.render(WINDOW,"trajectory 2",i) 
            score += reward
            agent.save_transition(observation, action, reward)
            observation = observation_
            count += 1
            env.render(WINDOW,title="trajectory 2")
            if (game_over == True):
              break
        print("this is the count 1 ",count)
       
        reward_loss = agent.train()
        cross_entropy.append([i,reward_loss])
        print('episode ', i, 'score %.2f' % score)
        sleep(0.3)