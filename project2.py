import numpy as np
import gym
from gym import wrappers
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
import keras
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import pandas as pd
#from keras import backend as K

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED']=str(0)


class DQN:
    def __init__(self, env):
        self.env = env
        self.obs_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 1.0  
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.memory = deque(maxlen=1000000)

        self.model1, self.model2, self.model3, self.model4, self.model5 = self.build_models()
        self.model = self.build_best_model()
        self.dup_model = self.model
        self.gamma1 = self.model
        self.gamma2 = self.model
        self.gamma3 = self.model
        self.gamma4 = self.model
        self.decay1 = self.model
        self.decay2 = self.model
        self.decay3 = self.model
        self.decay4 = self.model

    def build_best_model(self):
        # BEST MODEL WAS FOUND ALREADY WITH model_testing()... below units are from the best model performance
        inputs = Input(shape=(self.obs_space,))

        hidden_1 = Dense(256, activation='relu')(inputs)
        hidden_2 = Dense(128, activation='relu')(hidden_1)
        outputs = Dense(self.action_space, activation='linear')(hidden_2)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def build_models(self):
        models = []
        model_units = [[256, 128], [256, 256], [512, 128], [512, 256], [512, 64]]

        for units in model_units:
            inputs = Input(shape=(self.obs_space,))

            hidden_1 = Dense(units[0], activation='relu')(inputs)
            hidden_2 = Dense(units[1], activation='relu')(hidden_1)
            outputs = Dense(self.action_space, activation='linear')(hidden_2)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            models.append(model)
        return models[0], models[1], models[2], models[3], models[4]

    def getAction(self, obs, model):
        if np.random.rand() < self.epsilon:
            # Randomly explore
            return random.randrange(self.action_space)
        else:
            # Select action based on argmax
            return np.argmax(model.predict(obs)[0])

    def replay(self, rewards, model):
        # Wait for enough memory to accumulate
        if len(self.memory) < 32: 
            return

        batch = random.sample(self.memory, 32)

        # Used numpy vectorization for a speed-up in performance for creating batches
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])
        done_mask = 1 - dones
    
        # Predict on the batches pulled
        y_preds = rewards + (self.gamma * (np.max(model.predict_on_batch(np.squeeze(next_states)), axis=1))) * done_mask
        y_pred_batch = model.predict_on_batch(np.squeeze(states))

        batch_pos = np.array([i for i in range(32)])
        y_pred_batch[[batch_pos], [actions]] = y_preds

        # Fit the model on batches
        model.fit(np.squeeze(states), y_pred_batch, epochs=1, verbose=0)

    def decay_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        del self.memory
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 1.0  
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.memory = deque(maxlen=1000000)

    def model_testing(self):
        with open("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/Model_testing_log.txt", "w+") as f:
            f.write("Model Testing:\n\n")
            model_rewards = []
            for model in [self.model2, self.model3, self.model4, self.model5]:
                self.reset()
                rewards = []
                last_episode = 0
                n_episodes = 1000

                for episode in range(n_episodes):
                    t_step = 0
                    obs = self.env.reset()
                    obs = np.reshape(obs, (1, 8))
                    episode_reward = 0

                    while True:
                        env.render()
                        action = self.getAction(obs, model)
                        self.decay_epsilon()
                        next_obs, reward, done, _ = self.env.step(action)
                        next_obs = np.reshape(next_obs, (1, 8))
                        #print(reward)
                        episode_reward += reward
                        self.memory.append((obs, action, reward, next_obs, done))
                        obs = next_obs
                        self.replay(rewards, model)
                        if done:
                            f.write("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                            if episode % 10 == 0:
                                print("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                            break
                        t_step += 1
                    rewards.append(episode_reward)

                    # Average score of last 100 episodes
                    if np.mean(rewards[-100:]) > 200:
                        f.write("\n Experiment completed in {} episodes \n".format(episode))
                        print('\n Experiment completed in {} episodes \n'.format(episode))
                        break

                    f.write("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))
                    if episode % 10 == 0:
                        print("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))

                    last_episode = episode

                print("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                f.write("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                model_rewards.append(rewards)

                # window_size = 3
                # reward_series = pd.Series(rewards)
                # windows = reward_series.rolling(window_size)
                # rolling_avg = windows.mean().tolist()

            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(model_rewards)):
                ax.plot(range(len(model_rewards[i])), model_rewards[i], label="Model {}".format(i))
            #plt.plot(np.arange(len(rewards)), rolling_avg)
            legend = ax.legend(loc='lower right')
            plt.ylabel('Reward')
            plt.xlabel('Episodes')
            plt.title("Performance Comparison of Different Model Architectures")
            #plt.show()
            plt.savefig("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/multi_model_fig.png")

    def trainingGraph(self):
        with open("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/training_graph.txt", "w+") as f:
            f.write("Training Graph:\n\n")
            model = self.model1
            self.reset()
            rewards = []
            last_episode = 0
            n_episodes = 1000 # large enough to ensure ample opportunity to solve within a reasonable amount of time

            for episode in range(n_episodes):
                t_step = 0
                obs = self.env.reset()
                obs = np.reshape(obs, (1, 8))
                episode_reward = 0

                while True:
                    env.render()
                    action = self.getAction(obs, model)
                    self.decay_epsilon()
                    next_obs, reward, done, _ = self.env.step(action)
                    next_obs = np.reshape(next_obs, (1, 8))
                    #print(reward)
                    episode_reward += reward
                    self.memory.append((obs, action, reward, next_obs, done))
                    obs = next_obs
                    self.replay(rewards, model)
                    if done:
                        f.write("Episode {} finished after {} timesteps with reward {}.\n".format(episode, t_step+1, episode_reward))
                        if episode % 10 == 0:
                            print("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                        break
                    t_step += 1
                rewards.append(episode_reward)

                # Average score of last 100 episodes
                if np.mean(rewards[-100:]) > 200:
                    f.write("\n Experiment completed in {} episodes \n".format(episode))
                    print('\n Experiment completed in {} episodes \n'.format(episode))
                    break

                f.write("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))
                if episode % 10 == 0:
                    print("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))

                last_episode = episode

            print("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
            f.write("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))

            # Save trained model
            model.save("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/best_trained_model.h5")

            window_size = 3
            reward_series = pd.Series(rewards)
            windows = reward_series.rolling(window_size)
            rolling_avg = windows.mean().tolist()

            fig, ax = plt.subplots(figsize=(10, 5))
            plt.plot(np.arange(len(rewards)), rewards)
            plt.plot(np.arange(len(rewards)), rolling_avg)
            plt.ylabel('Reward')
            plt.xlabel('Episodes')
            #plt.show()
            plt.savefig("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/training_graph.png")

    def testingGraph(self):
        with open("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/testing_graph.txt", "w+") as f:
            f.write("Testing Graph:\n\n")
            model = keras.models.load_model("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/best_trained_model.h5")
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            self.reset()
            rewards = []
            last_episode = 0
            n_episodes = 100

            for episode in range(n_episodes):
                t_step = 0
                obs = self.env.reset()
                obs = np.reshape(obs, (1, 8))
                episode_reward = 0

                while True:
                    env.render()
                    action = self.getAction(obs, model)
                    self.decay_epsilon()
                    next_obs, reward, done, _ = self.env.step(action)
                    next_obs = np.reshape(next_obs, (1, 8))
                    #print(reward)
                    episode_reward += reward
                    self.memory.append((obs, action, reward, next_obs, done))
                    obs = next_obs
                    self.replay(rewards, model)
                    if done:
                        f.write("Episode {} finished after {} timesteps with reward {}.\n".format(episode, t_step+1, episode_reward))
                        if episode % 10 == 0:
                            print("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                        break
                    t_step += 1
                rewards.append(episode_reward)

            print("Total reward received after 100 episodes: {}".format(sum(rewards)))
            f.write("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))

            window_size = 3
            reward_series = pd.Series(rewards)
            windows = reward_series.rolling(window_size)
            rolling_avg = windows.mean().tolist()

            fig, ax = plt.subplots(figsize=(10, 5))
            plt.plot(np.arange(len(rewards)), rewards)
            plt.plot(np.arange(len(rewards)), rolling_avg)
            plt.ylabel('Reward')
            plt.xlabel('Episodes')
            #plt.show()
            plt.savefig("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/testing_graph.png")

    def gammaGraph(self):
        with open("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/gammas_graph.txt", "w+") as f:
            f.write("Gamma Testing:\n\n")
            self.reset()
            gammas = [0.99, 0.999, 1.0]
            gamma_models = [self.gamma2, self.gamma3, self.gamma4]
            model_rewards = []
            for i, gamma in enumerate(gammas):
                model = gamma_models[i]
                self.gamma = gamma
                rewards = []
                last_episode = 0
                n_episodes = 1000

                for episode in range(n_episodes):
                    t_step = 0
                    obs = self.env.reset()
                    obs = np.reshape(obs, (1, 8))
                    episode_reward = 0

                    while True:
                        env.render()
                        action = self.getAction(obs, model)
                        self.decay_epsilon()
                        next_obs, reward, done, _ = self.env.step(action)
                        next_obs = np.reshape(next_obs, (1, 8))
                        #print(reward)
                        episode_reward += reward
                        self.memory.append((obs, action, reward, next_obs, done))
                        obs = next_obs
                        self.replay(rewards, model)
                        if done:
                            f.write("Episode {} finished after {} timesteps with reward {}.\n".format(episode, t_step+1, episode_reward))
                            if episode % 10 == 0:
                                print("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                            break
                        t_step += 1
                    rewards.append(episode_reward)

                    # Average score of last 100 episodes
                    if np.mean(rewards[-100:]) > 200:
                        f.write("\n Experiment completed in {} episodes \n".format(episode))
                        print('\n Experiment completed in {} episodes \n'.format(episode))
                        break

                    f.write("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))
                    if episode % 10 == 0:
                        print("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))

                    last_episode = episode

                print("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                f.write("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                model_rewards.append(rewards)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(model_rewards)):
                ax.plot(range(len(model_rewards[i])), model_rewards[i], label="Gamma {}".format(gammas[i]))
            #plt.plot(np.arange(len(rewards)), rolling_avg)
            legend = ax.legend(loc='lower right')
            plt.ylabel('Reward')
            plt.xlabel('Episodes')
            plt.title("Training Comparison of Gamma")
            #plt.show()
            plt.savefig("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/gammas_graph.png")

    def learningRateGraph(self):
        with open("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/learning_rates_graph.txt", "w+") as f:
            f.write("Learning Rate Testing:\n\n")
            self.reset()
            lrs = [0.01, 0.001, 0.0001]
            lr_models = []
            for lr in lrs:
                inputs = Input(shape=(self.obs_space,))

                hidden_1 = Dense(256, activation='relu')(inputs)
                hidden_2 = Dense(128, activation='relu')(hidden_1)
                outputs = Dense(self.action_space, activation='linear')(hidden_2)

                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
                lr_models.append(model)
            
            model_rewards = []
            for model in lr_models:
                self.reset()
                rewards = []
                last_episode = 0
                n_episodes = 500

                for episode in range(n_episodes):
                    t_step = 0
                    obs = self.env.reset()
                    obs = np.reshape(obs, (1, 8))
                    episode_reward = 0

                    while True:
                        env.render()
                        action = self.getAction(obs, model)
                        self.decay_epsilon()
                        next_obs, reward, done, _ = self.env.step(action)
                        next_obs = np.reshape(next_obs, (1, 8))
                        #print(reward)
                        episode_reward += reward
                        self.memory.append((obs, action, reward, next_obs, done))
                        obs = next_obs
                        self.replay(rewards, model)
                        if done:
                            f.write("Episode {} finished after {} timesteps with reward {}.\n".format(episode, t_step+1, episode_reward))
                            if episode % 10 == 0:
                                print("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                            break
                        t_step += 1
                    rewards.append(episode_reward)

                    # Average score of last 100 episodes
                    if np.mean(rewards[-100:]) > 200:
                        f.write("\n Experiment completed in {} episodes \n".format(episode))
                        print('\n Experiment completed in {} episodes \n'.format(episode))
                        break

                    f.write("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))
                    if episode % 10 == 0:
                        print("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))

                    last_episode = episode

                print("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                f.write("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                model_rewards.append(rewards)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(model_rewards)):
                ax.plot(range(len(model_rewards[i])), model_rewards[i], label="Learning Rate {}".format(lrs[i]))
            #plt.plot(np.arange(len(rewards)), rolling_avg)
            legend = ax.legend(loc='lower right')
            plt.ylabel('Reward')
            plt.xlabel('Episodes')
            plt.title("Training Comparison of Learning Rates")
            #plt.show()
            plt.savefig("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/learning_rates_graph.png")

    def epsilonDecayGraph(self):
        with open("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/epsilon_decay_graph.txt", "w+") as f:
            f.write("Epsilon Decay Testing:\n\n")
            self.reset()
            decays = [0.9, 0.99, 0.999, 0.9999]
            #decay_models = [self.decay1, self.decay2, self.decay3, self.decay4]
            model_rewards = []
            for i, decay in enumerate(decays):
                model = self.build_best_model()
                self.epsilon_decay = decay
                rewards = []
                last_episode = 0
                n_episodes = 500

                for episode in range(n_episodes):
                    t_step = 0
                    obs = self.env.reset()
                    obs = np.reshape(obs, (1, 8))
                    episode_reward = 0

                    while True:
                        env.render()
                        action = self.getAction(obs, model)
                        self.decay_epsilon()
                        next_obs, reward, done, _ = self.env.step(action)
                        next_obs = np.reshape(next_obs, (1, 8))
                        #print(reward)
                        episode_reward += reward
                        self.memory.append((obs, action, reward, next_obs, done))
                        obs = next_obs
                        self.replay(rewards, model)
                        if done:
                            f.write("Episode {} finished after {} timesteps with reward {}.\n".format(episode, t_step+1, episode_reward))
                            if episode % 10 == 0:
                                print("Episode {} finished after {} timesteps with reward {}.".format(episode, t_step+1, episode_reward))
                            break
                        t_step += 1
                    rewards.append(episode_reward)

                    # Average score of last 100 episodes
                    if np.mean(rewards[-100:]) > 200:
                        f.write("\n Experiment completed in {} episodes \n".format(episode))
                        print('\n Experiment completed in {} episodes \n'.format(episode))
                        break

                    f.write("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))
                    if episode % 10 == 0:
                        print("Average reward over the last 100 episodes: {0:.2f} \n".format(np.mean(rewards[-100:])))

                    last_episode = episode

                print("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                f.write("Total reward received after {} episodes: {}".format(last_episode, sum(rewards)))
                model_rewards.append(rewards)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(model_rewards)):
                ax.plot(range(len(model_rewards[i])), model_rewards[i], label="Epsilon Decay {}".format(decays[i]))
            #plt.plot(np.arange(len(rewards)), rolling_avg)
            legend = ax.legend(loc='lower right')
            plt.ylabel('Reward')
            plt.xlabel('Episodes')
            plt.title("Training Comparison of Epsilon Decay Rate")
            #plt.show()
            plt.savefig("C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/epsilon_decay_graph.png")

if __name__ == "__main__":
    #env = wrappers.Monitor(env, 'C:/Users/gabri/Desktop/python_work/OMSCS-Reinforcement_Learning/lunarlander-experiment', video_callable=lambda episode_id: Ture, force=True)
    dqn = DQN(env)
    #dqn.model_testing() # comment this out once you've tested for best model
    #exit()

    # Best model obtained, now run training and tests using best model
    dqn.trainingGraph()
    dqn.testingGraph()
    dqn.gammaGraph()
    dqn.learningRateGraph()
    dqn.epsilonDecayGraph()