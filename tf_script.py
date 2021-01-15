#!/usr/bin/env python
# coding: utf-8

#13/01/2020

# pip install pandas
# pip install tqdm
# pip install scikit-learn

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from os import listdir
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle
from collections import deque
import time
import random
#from tqdm import tqdm
import os

print("tensorflow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) 

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training (CHANGE BACK TO ~50_000)
MIN_REPLAY_MEMORY_SIZE = 5_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '256_512_512_256'
MIN_REWARD = -100 # For model save
MEMORY_FRACTION = 0.20
OBSERVATION_WINDOW = 30
EPISODE_STEPS = 200


# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

FEE = 20

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

### build Episodes

DATA_FILE_SAMPLES = 50
# EPISODES = 1001

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def detrend(data):
    #removes trend from timeseries data
    data = pd.DataFrame(data)
    DATA = data.diff()
    DATA = DATA.iloc[1:]
    DATA = pd.DataFrame.to_numpy(DATA)
    return DATA

def episode_window(data):
    #returns a list of episode_windows
    episode_data_window = EPISODE_STEPS + OBSERVATION_WINDOW
    windows = []

    for e in range(random.randint(0,30), len(data)-episode_data_window, random.randint(1,50)):
        #returns a list of observation_windows
        window_start = e
        window_end = e + episode_data_window
        window = data[window_start:window_end]
        windows.append(window)

    return windows

def observation_window(window):
    #returns a list of tuple of (observation_window and current_price)
    observation_windows = [] 
    
    for e in range(EPISODE_STEPS):
        window_start = e
        window_end = e + OBSERVATION_WINDOW + 1
        observation_window = window[window_start:window_end].to_numpy()
           
        if e < EPISODE_STEPS-1:
            current_price = window.iloc[e + OBSERVATION_WINDOW][0]
        else:
            current_price = window.iloc[e + OBSERVATION_WINDOW][3]
        
        observation_window = detrend(observation_window)
        observation_window = scaler.transform(observation_window)
        
        output = (observation_window, current_price)
       
        observation_windows.append(output)
        
    return observation_windows

# for paperspace instance
path = r'/storage/data/'
# path = r'C:\Users\Tristan\Desktop\github_asx_training_repo\asx_data'


files = find_csv_filenames(path)

print("number of samples: " , len(files))

original_files = shuffle(files)
print('shuffled files: ' ,len(original_files))
files = original_files[0:DATA_FILE_SAMPLES]

master_data = []

file_counter = 0

for file in files:
#     print("file: ", file)
    file_counter += 1
    file_path = os.path.join(path, file)
    DATA = pd.read_csv(file_path)
    DATA.dropna(inplace=True)
    DATA.drop(['Date','Adj Close'], axis=1, inplace=True)
    DATA.astype('float')
    
    scaler = preprocessing.StandardScaler().fit(DATA)
    
    data = episode_window(DATA)

    all_data = []
    counter = 0
    for i in range(len(data)):
        counter += 1
        cob = observation_window(data[i])
        all_data.append(cob)
    print(file_counter, "/", DATA_FILE_SAMPLES, " file: ", file, len(all_data), "windows")
        
    master_data.extend(all_data)
    

shuffled_data = shuffle(master_data, random_state=0)

# Environment settings
# EPISODES = len(shuffled_data[0:EPISODES])

print("Shuffled Data Len: ", len(shuffled_data))

EPISODES = len(shuffled_data)

print("Number of Episodes: ", EPISODES)



class Trader:
    
    def __init__ (self):
        self.price = 2.5
        self.kitty = 3000 #will be initilize at the start of an episode and update throughout the episode
        self.fee = FEE
        self.volume = 0
        self.current_value = 0 
        self.purchase_price = 0 
        self.reward = 0
        
    def action(self, choice): 
        '''
        Gives 3 trade options. (0 = Buy, 1 = Hold, 2 = Sell)
        '''
        if choice == 0:
            if self.volume > 0: #if there is no money in the kitty to purchase volume, pass
                self.reward = -1
                self.volume = self.volume
                self.current_value = self.current_value
                self.purchase_price = self.purchase_price
                self.kitty = self.kitty

            else:
                #calculate number of shares to buy
                self.volume = (self.kitty- self.fee) // env.price()
                
                #calculate the estimate value of the kitty after trade
                self.current_value = (self.volume * env.price()) - self.fee
            
                #set purchase price
                self.purchase_price = self.current_value
            
                #reduce available kitty after purchase
                self.kitty = self.kitty - (self.fee + self.current_value)

                self.reward = -1

                #return self.current_value, self.volume, self.kitty, self.reward
        
        if choice == 1:
            #hold position
            if self.volume == 0: 
                self.reward = -1
                self.volume = self.volume
                self.current_value = self.current_value
                self.purchase_price = self.purchase_price
                self.kitty = self.kitty
            else:
                self.purchase_price = self.purchase_price
                self.volume = self.volume 
                self.kitty = self.kitty
                self.current_value = self.volume * env.price()
                self.reward = (self.current_value - self.purchase_price) / self.purchase_price

            #return self.current_value, self.volume, self.kitty, self.reward
                   
        if choice == 2:
            if self.volume <= 0: #if there is volume to sell, pass 
                self.reward = -1
                self.volume = self.volume
                self.current_value = self.current_value
                self.purchase_price = self.purchase_price
                self.kitty = self.kitty
            else: #if there is money in the kitty, continue with trade (sell all of the available position)
                #calculate the estimated value of the kitty after trade
                self.sale_value = self.volume * env.price()

                #restock the kitty following the trade
                self.kitty = self.sale_value + self.kitty
                
                #calculate reward
                self.reward = (self.sale_value - self.purchase_price) / self.purchase_price
                
                #reset current value to 0
                self.current_value = 0
                
                #reset volume to 0
                self.volume = 0
                
            #return self.current_value, self.volume, self.kitty, self.reward 
        # print("Choice: ", choice, "Reward: ", self.reward, "Volume: ", self.volume, "Price: ", env.price(), "Kitty: ", self.kitty, "Current Value: ", self.current_value)    
    
    def current_value(self, current_value):
    	return self.current_value 

    def volume(self, volume):
    	return self.volume
        
    def kitty(self, kitty):
        return self.kitty

    def reward(self):
        return self.reward

class MarketEnv:
    TIME_STEP_PENALTY = -1 
    OBSERVATION_SPACE_VALUES = (OBSERVATION_WINDOW, 5)
    ACTION_SPACE_SIZE = 3

    def __init__(self):
        self.trader = Trader()

    def reset(self):
        self.episode_step = 0
        self.trader.current_value = 0
        self.trader.volume = 0
        self.trader.kitty = 30000
   
    def step(self, action, reward=-1):
        self.trader.action(action)
        
        new_observation = shuffled_data[episode-1][self.episode_step][0]
        
        reward = self.trader.reward
       
        if self.episode_step >= 199:
            done = True
        else:
            done = False

        self.episode_step += 1

        return new_observation, reward, done

    def price(self):
        price = shuffled_data[episode-1][self.episode_step][1]
        return price

env = MarketEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('/artifacts/models'):
    os.makedirs('/artifacts/models')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
       # self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
            self.step += 1
            self.writer.flush()

    _train_dir = os.path.dirname(os.path.realpath('/artifacts/models'))
    _log_write_dir = os.path.dirname(os.path.realpath('/artifacts/models'))
    _should_write_train_graph = False

    def _train_step(self):
        pass

#Agent class

class DQNAgent:
    def __init__(self):
       
        #Main Model - Train this model every step
        self.model = self.create_model()
       
        #Target Model - Predict this model every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
       
        #An array with the last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
       
        #Custom TensorBoard object
        self.tensorboard = ModifiedTensorBoard(log_dir="/storage/logs/{}-{}".format(MODEL_NAME, int(time.time())))
       
        #Uesd to count when time to update target model with main model weights
        self.target_update_counter = 0
       
    def create_model(self):
        
#         try:
        model = tf.keras.models.load_model(r'/storage/models/256_512_512_256_____2.53max_-106.63avg_-184.02min__1610647079.model')
        print("model = storage/models/256_512_512_256_____2.53max_-106.63avg_-184.02min__1610647079.model")
#         else:
#             model = Sequential()
#             model.add(Dense(150, input_shape=env.OBSERVATION_SPACE_VALUES))
#             model.add(Activation('relu'))
#             model.add(Flatten())

#             model.add(Dense(256))
#             model.add(Activation('relu'))
#             model.add(Dropout(.2))

#             model.add(Dense(512))
#             model.add(Activation('relu'))
#             model.add(Dropout(.2))

#             model.add(Dense(256))
#             model.add(Activation('relu'))
#             model.add(Dropout(.2))

#             model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear')) #ACTION_SPACE_SIZE = how many choice (3)
#             model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
#             print("model = scratch")
        return model

    #Adds step's data to a memory replay array
    #(observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
 
    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        if np.array([transition[0] for transition in minibatch]).shape == (32, 30, 5):
            current_states = np.array([np.array(transition[0]) for transition in minibatch])
        else:
            while np.array([transition[0] for transition in minibatch]).shape != (32, 30, 5):
                minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
                current_states = np.array([np.array(transition[0]) for transition in minibatch])      

        # current_states = tf.convert_to_tensor(current_states, dtype=tf.float32)
        # print("current_states type: ", type(current_states))
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    #Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)) ###255 to be repaced with normalizing function

agent = DQNAgent()

# Iterate over episodes
# for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
for episode in range(EPISODES):
    
    start_time = time.time()

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            try:
                action = np.argmax(agent.get_qs(current_state))
            except:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)


        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        #if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
#             env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1      

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #portfolio_value = 
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'/artifacts/models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        if episode%500 == 0:
            agent.model.save(f'/artifacts/models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    end_time = time.time()
    print(episode, " of ", EPISODES, " complete...", (end_time - start_time))

