import numpy as np
import tensorflow as tf
# from tensorflow import keras

import tensorflow_probability as tfp

from configs import *

class Agent:
    
    def __init__(self, world_model=None):
        self.config = lower_level_config
        self.hyper_layers = lower_level_config["hypernet_layers"]
        self.policy_layers = lower_level_config["policy_layers"]
        self.base_timesteps = lower_level_config["base_timesteps"]
        self.hyper_inputs = lower_level_config["hyper_input_size"]
        self.policy_ip_sz = lower_level_config["policy_input_size"]
        self.policy_op_sz = lower_level_config["policy_output_size"]
        self.temperature = lower_level_config["action_logits_temperature"]
        self.l2_lambda = lower_level_config["l2_lambda"]
        self.steps_threshold = lower_level_config["steps_threshold"]
        self.action_map = lower_level_config["action_mapping"]
        
        self.higher_layers = higher_level_config["action_net_layers"]
        self.higher_input = higher_level_config["higher_input_size"]
        self.higher_output = higher_level_config["higher_output_size"]
        self.abstract_timestamps = higher_level_config["abstract_timestamps"]
        
        
        self.epoch_rewards = []
        self.abstract_reward = []
        
        # Initialize optimizer, networks and parameters
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lower_level_config["lr"])
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=lower_level_config["b_lr"])
        self.abstract_optimizer = tf.keras.optimizers.Adam(learning_rate=lower_level_config["lr"])
        
        self.trainable_layers = []
        self.baseline_layers = []
        self.abstract_layers = []
        
        self._abstract()
        self._hypernet()
        self._baseline()
        
        # If world model is trained, use it to do RL and planning
        # If wm == None => Use environment directly
        self.wm = world_model
        
    
    def test_step(self, env, start_position):
        '''
        Mostly similar to forward pass
        '''
        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"########     TEST STEP  {env.env_name} #########")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"\nGoal : {env.goal}, Start : {start_position}")
        
        action_vector = env.valid_goals[env.goal]
        print("Action Vector : ", action_vector)
        
        higher_state_latent = tf.expand_dims(tf.convert_to_tensor(action_vector), axis=0)
        # print(higher_state_latent.shape)
        functional_parameters = self.generate_functional_params(higher_state_latent)
        functional = lambda x: self.construct_functional_from_params(x,
                    functional_parameters)
        
        state = env.reset(start_position)
        # env.print_board()  # Print current state?
        
        # Move in the grid but with MAX probability actions
        for i in range(self.base_timesteps):
            state = tf.cast(tf.convert_to_tensor(state), tf.float32)
            prev_state = env.state
            action_probs, logits = functional(state)
            
            # MAX action
            action = np.argmax(tf.stop_gradient(logits).numpy())
            state, reward, done = env.step(action)
            print("Current State : ", prev_state) 
            print("Action Probabilities : ", tf.nn.softmax(logits.numpy()/0.5).numpy()) 
            print("Action Taken : ", self.action_map[action]) 
            print("Reward : ", reward)
            
            
            # env.print_board()
            
            if done == 0:
                print("reached goal")
                break 
        
        env.print_policy_map()
    
    ####################################
    # TRAINING THE 2 LEVEL MODEL
    ####################################
            
    def train_step(self, env, epoch):
        
        reinforce_vars = {
            "log_probs":[],
            "mask":[],
            "reward":[],
            "l2_loss":[],
            "lower_baseline":[]
        }

        action_vector = env.valid_goals[env.goal]
        
        with tf.GradientTape() as tape:
            trainable_params = self.get_lower_action_vars()
            baseline_params = self.get_baseline_vars()
            
            tape.watch(trainable_params+baseline_params)
            
            reinforce_vars = self.forward_lower_level(action_vector, env, reinforce_vars, epoch)
            if len(reinforce_vars["reward"]) < self.steps_threshold:
                # Skip training if episode length is too small
                print("Skipping small episode ... ")
                return
            
            # Backprop with reinforce gradients
            rl_loss, baseline_loss = self.REINFORCE(reinforce_vars)

        gradients = tape.gradient(rl_loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
    
        gradients2 = tape.gradient(baseline_loss, baseline_params)
        gradients2, _ = tf.clip_by_global_norm(gradients2, 2.0)
        self.baseline_optimizer.apply_gradients(zip(gradients2, baseline_params))


    def abstract_train_step(self, X, y, wm):
        '''
        - wm is the World Model
        - Use wm to get transitions
        - use like this : wm.next_state()
        '''

        print("here trying to train abstract policy")
        
            
        abstract_vars = {
            "log_probs":[],
            "mask":[],
            "reward":[]   
        }
        with tf.GradientTape() as tape:
            abstract_params = self.get_abstract_params()
            tape.watch(abstract_params)
            
            reinforce_vars, reward = self.forward_higher_level(wm, abstract_vars, X, y)
            loss = self.REINFORCE_abstract(reinforce_vars)
            
            self.abstract_reward.append(reward)
        
        gradients = tape.gradient(loss, abstract_params)
        # print("################# gradients ############# \n", gradients)
        gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
        self.abstract_optimizer.apply_gradients(zip(gradients, abstract_params))
            
    
    def forward_higher_level(self, wm, vars, x, y):
        # print("here in forward pass")
        input, goal = tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32)
        print("Forward pass goal : ", goal)
        actions = []
        
        for i in range(self.abstract_timestamps):
            print(input)
            state_input = tf.expand_dims(tf.convert_to_tensor(tf.concat([input, goal], axis=0)), axis=0)
            action_probs = self.action_dense4(self.action_dense2(self.action_dense1(state_input)))
            
            dist = tfp.distributions.OneHotCategorical(probs=action_probs)
            action = tf.squeeze(tf.stop_gradient(dist.sample()))
            log_probs = dist.log_prob(action)
            actions.append(action.numpy())
            reward = self.abstract_rewards(actions, y)
            
            print("Abstract step : ", action_probs.numpy(), action.numpy(), \
                    log_probs.numpy(), reward)
            
            vars["log_probs"].append(log_probs)
            vars["reward"].append(reward)
            
            # Take a step
            input = wm.next_state(input, tf.cast(action, tf.float32))

            if reward > 0 :
                break
        
        return vars, reward

    def forward_lower_level(self, higher_state_latent, env, r_vars, epoch):
        # Since no batch, we reshape latent vector
        
        higher_state_latent = tf.expand_dims(tf.convert_to_tensor(higher_state_latent), axis=0)
        # print(higher_state_latent.shape)
        functional_parameters = self.generate_functional_params(higher_state_latent)
        functional = lambda x: self.construct_functional_from_params(x,
                    functional_parameters)
        
        state = env.reset()
        env.print_board()  # Print current state?
        print("Start State : ", env.state)
        
        for i in range(self.base_timesteps):
            
            
            state = tf.cast(tf.convert_to_tensor(state), tf.float32)
            
            _state = tf.expand_dims(state, axis=0)
            # print("_state : ", _state.shape)
            baseline_value = self.base2(self.base1(_state))
            r_vars["lower_baseline"].append(baseline_value)
            
            action_probs, logits = functional(state)
            distribution = tfp.distributions.Categorical(logits=logits)
            action = distribution.sample()
            prev_state = env.state
            # print("before taking a step : ", action.numpy(), state)
            state, reward, done = env.step(action.numpy())

            # print("after step : ", state)
            
            l2 = tf.reduce_sum(tf.square(logits))
            log_probs = distribution.log_prob(tf.stop_gradient(tf.squeeze(action)))
            
            r_vars["log_probs"].append(log_probs)
            r_vars["l2_loss"].append(l2)
            r_vars["reward"].append(reward)
            r_vars["mask"].append(1)
            
            if epoch % PRINT_ENV_PER_EPOCHS == 0 :
                print("Current State : ", prev_state) 
                print("Action Probabilities : ", tf.nn.softmax(logits.numpy()).numpy()) 
                print("Action Taken : ", self.action_map[action.numpy()]) 
                print("Reward : ", reward)
                env.print_board()
            
            if done == 0:
                print("reached goal")
                break
        
        print("Episode Rewards : ", sum(r_vars["reward"]))
        
        return r_vars



    #################################
    # REINFORCE policy gradients
    #################################
    
    def REINFORCE_abstract(self, vars):
        print("Trying to reinforce")
        rewards = tf.convert_to_tensor(vars["reward"])
        log_probs = tf.convert_to_tensor(vars["log_probs"])
        
        print("Rewards : ", rewards)
        print("Log probs : ", log_probs)
        
        GAMMA = 0.99
        R = 0
        eps = tf.cast(tf.convert_to_tensor(np.finfo(np.float32).eps.item()), tf.float32)
        returns = []
        # print("reinforce debug rewars: ", tf.transpose(rewards)[::-1])
        for r in tf.transpose(rewards)[::-1]:
            R = r.numpy() + GAMMA * R
            returns.insert(0, R)
        
        returns = tf.cast(tf.convert_to_tensor(returns), tf.float32)
        discounted_returns = (returns - tf.expand_dims(tf.reduce_mean(returns), axis=0)) \
                            / (tf.expand_dims(tf.math.reduce_std(returns), axis=0) + eps)
        
        reinforce_loss = -tf.reduce_sum(log_probs * discounted_returns)
        
        print("Reinforce Loss : ", reinforce_loss.numpy())
        return reinforce_loss
        
        
        
    def REINFORCE(self, vars):
        rewards = tf.convert_to_tensor(vars["reward"])
        self.epoch_rewards.append(sum(vars["reward"]))
        l2_loss = tf.reduce_mean(tf.convert_to_tensor(vars["l2_loss"]))
        mask = tf.cast(tf.convert_to_tensor(vars["mask"]), tf.float32)
        log_probs = tf.convert_to_tensor(vars["log_probs"])
        baseline_values = tf.convert_to_tensor(vars["lower_baseline"])

  
        # print("baseline : ", baseline_values.shape)
        GAMMA = 0.99
        R = 0
        eps = tf.cast(tf.convert_to_tensor(np.finfo(np.float32).eps.item()), tf.float32)
        returns = []
        # print("reinforce debug rewars: ", tf.transpose(rewards)[::-1])
        for r in tf.transpose(rewards)[::-1]:
            R = r.numpy() + GAMMA * R
            returns.insert(0, R)
        
        returns = tf.cast(tf.convert_to_tensor(returns), tf.float32)
        # print("returns : ", returns) 
        
        discounted_returns = (returns - tf.expand_dims(tf.reduce_mean(returns), axis=0)) \
                            / (tf.expand_dims(tf.math.reduce_std(returns), axis=0) + eps)
                            
                            
        baseline_mse = tf.reduce_mean(tf.square(discounted_returns - tf.squeeze(baseline_values)))
        discounted_returns = discounted_returns - tf.squeeze(tf.stop_gradient(baseline_values))
        
        
        
        # discounted_returns = returns                  
        # print("discounted returns after baseline reduction: ", discounted_returns)
        # print("Log probs before : ", log_probs.shape, mask.shape)
        # print("before mask : ", log_probs, mask)
        
        log_probs = log_probs * mask
        
        # print("after_mask", log_probs)
        # print("log probs masked : ", log_probs.shape)
        # print("discounting : ", log_probs.shape, discounted_returns.shape)
        reinforce_loss = tf.reduce_sum(log_probs * discounted_returns)
        total_loss = - reinforce_loss + self.l2_lambda * l2_loss
        
        print("LOSS : ", reinforce_loss.numpy(), self.l2_lambda*l2_loss.numpy(), baseline_mse.numpy())
        
        return total_loss, baseline_mse



    ###################################
    # Higher Action network Functions
    ###################################
    
    def _abstract(self):
        
        self.action_dense1 = tf.keras.layers.Dense(self.higher_layers[0], activation=tf.nn.leaky_relu)
        self.action_dense2 = tf.keras.layers.Dense(self.higher_layers[1], activation=tf.nn.leaky_relu)
        # self.action_dense3 = tf.keras.layers.Dense(self.higher_layers[2], activation=tf.nn.leaky_relu)
        self.action_dense4 = tf.keras.layers.Dense(self.higher_output, activation=tf.nn.softmax)
        
        self.abstract_layers.append(self.action_dense1)
        self.abstract_layers.append(self.action_dense2)
        # self.abstract_layers.append(self.action_dense3)
        self.abstract_layers.append(self.action_dense4)
    
    def get_abstract_params(self):
        vars = []
        
        for layer in self.abstract_layers:
            vars += layer.trainable_variables
        
        # print("Vars : ", vars)
        return vars
    
    def abstract_rewards(self, action_seq, correct_seq):
        print("Giving reward based on sequence")
        size = min(len(correct_seq), len(action_seq))
        # TODO
        print("Sequences : ", action_seq, correct_seq)
        
        flag = 1
        for i in range(size):
            print(np.argmax(action_seq[i]), np.argmax(correct_seq[i]))
            if np.argmax(action_seq[i]) != np.argmax(correct_seq[i]):
                flag = -1
        
        return flag
    
    
    ############################
    # HYPERNET FUNCTIONS
    ############################
    
    def _hypernet(self):
        self.hyp1 = tf.keras.layers.Dense(self.hyper_layers[0],name='ha1',activation=tf.nn.leaky_relu)
        self.hyp2 = tf.keras.layers.Dense(self.hyper_layers[1],name='ha2',activation=tf.nn.leaky_relu)
        self.hyp3 = tf.keras.layers.Dense(self.hyper_layers[2],name='ha2',activation=tf.nn.leaky_relu)
        
        self.w1 = tf.keras.layers.Dense(self.policy_ip_sz*self.policy_layers[0],name='w1', activation=tf.nn.relu, \
                                        kernel_initializer=hyperfanin_for_kernel(self.policy_ip_sz*self.policy_layers[0]))
        self.b1 = tf.keras.layers.Dense(self.policy_layers[0], activation=tf.nn.relu, \
                                        kernel_initializer=hyperfanin_for_bias())
        
        self.w2 = tf.keras.layers.Dense(self.policy_layers[0]*self.policy_layers[1],name='w2',activation=tf.nn.relu, \
                                        kernel_initializer=hyperfanin_for_kernel(self.policy_ip_sz*self.policy_layers[0]))
        self.b2 = tf.keras.layers.Dense(self.policy_layers[1], activation=tf.nn.relu, \
                                        kernel_initializer=hyperfanin_for_bias())
        
        self.w3 = tf.keras.layers.Dense(self.policy_layers[1]*self.policy_op_sz,name='w3',activation=tf.nn.relu, \
                                        kernel_initializer=hyperfanin_for_kernel(self.policy_ip_sz*self.policy_layers[0]))
        self.b3 = tf.keras.layers.Dense(self.policy_op_sz, activation=tf.nn.relu, \
                                        kernel_initializer=hyperfanin_for_bias())
        
        self.trainable_layers.append(self.hyp1)
        self.trainable_layers.append(self.hyp2)
        # self.trainable_layers.append(self.hyp3)
        self.trainable_layers.append(self.w1)
        self.trainable_layers.append(self.b1)
        self.trainable_layers.append(self.w2)
        self.trainable_layers.append(self.b2)
        self.trainable_layers.append(self.w3)
        self.trainable_layers.append(self.b3)
        
    def generate_functional_params(self, hidden_action_vector):
        # print("Generating hypernet ...")
        # Forward pass to generate the funtional kernels and bias
        hidden_vector = self.hyp3(self.hyp2(self.hyp1(hidden_action_vector)))
    
        w1 = self.w1(hidden_vector)
        b1 = self.b1(hidden_vector)
        w2 = self.w2(hidden_vector)
        b2 = self.b2(hidden_vector)
        w3 = self.w3(hidden_vector)
        b3 = self.b3(hidden_vector)

        # return layerwise params
        return [[w1, b1], [w2, b2], [w3, b3]]

    def construct_functional_from_params(self, state_input, 
                                        functional_parameters):
        # Extract weights
        w1, b1 = functional_parameters[0]
        w2, b2 = functional_parameters[1]
        w3, b3 = functional_parameters[2]

        # print("inputs in construct functional : ", state_input.shape, action_input.shape)
        # inputs = tf.concat([state_input, action_input], axis=1)
        # print("final inputs shape : ", inputs.shape)
        
        # Reshape for matmul
        inputs = tf.reshape(state_input, (-1, 1, self.policy_ip_sz))
        w1 = tf.reshape(w1, (-1, self.policy_ip_sz, self.policy_layers[0]))
        b1 = tf.reshape(b1, (-1, 1, self.policy_layers[0]))
        w2 = tf.reshape(w2, (-1, self.policy_layers[0], self.policy_layers[1]))
        b2 = tf.reshape(b2, (-1, 1, self.policy_layers[1]))
        w3 = tf.reshape(w3, (-1, self.policy_layers[1], self.policy_op_sz))
        b3 = tf.reshape(b3, (-1, self.policy_op_sz))

        # Actual neural network operations
        hidden1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
        logits = tf.matmul(hidden2, w3) + b3
        # print(logits.numpy())
        probs = tf.nn.softmax(logits/self.temperature)
        # print("outputs : ", output.shape)
        return tf.squeeze(probs), tf.squeeze(logits)
    
    def get_lower_action_vars(self):
        var_list = []
        for layer in self.trainable_layers:
            var_list += layer.trainable_variables
        
        # print("agent vars : ", var_list)
        return var_list

    
    ######################################
    # BASELINE VARIABLES
    ######################################

    def _baseline(self):
        self.base1 = tf.keras.layers.Dense(32,name='base1',activation=tf.nn.leaky_relu)
        self.base2 = tf.keras.layers.Dense(1,name='base2',activation=tf.nn.leaky_relu)
        
        self.baseline_layers.append(self.base1)
        self.baseline_layers.append(self.base2)
        
    def get_baseline_vars(self):
        var_list = []
        for layer in self.baseline_layers:
            var_list += layer.trainable_variables
        # print(var_list)
        return var_list
    