'''
Objectives :-

* Train (s_t, a_t) -> s_t+1
* Get a plot of loss

'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE

from dataloader import *
from models.state_model_hypernet import LinearModule, NonLinearModule, RecurrentModule
from models.learnable_story import LearnableStory

LINEAR_LEARNING_RATE = 0.01
LINEAR_EPOCHS = 500

NONLINEAR_LEARNING_RATE = 0.01
NONLINEAR_EPOCHS = 500

RECURRENT_LEARNING_RATE = 0.01
RECURRENT_EPOCHS = 500

LR1 = 0.01
LR2 = 0.05
HYPER_EPOCHS = 500

# Data stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data1, data2 = get_transition_data()
# data = data1+data2
# random.shuffle(data)

# X = torch.from_numpy(np.array([np.concatenate((k[0], k[1])) for k in data])).type(torch.FloatTensor).to(device)
# y = torch.from_numpy(np.array([k[2] for k in data])).type(torch.FloatTensor).to(device)

# print("X : ", X.shape, "Y : ", y.shape)

#############################
# Training a Linear Model
#############################

# model1 = LinearModule().to(device)
# optimizer1 = optim.Adam(model1.parameters(), LINEAR_LEARNING_RATE)

# Put model in train mode
model1.train()
loss1 = []

for epoch in range(LINEAR_EPOCHS):
    # Batch training
    optimizer1.zero_grad()
    output = model1(X)
    loss = F.mse_loss(output, y)
    loss1.append(loss.detach().cpu().numpy())
    loss.backward()
    optimizer1.step()



#############################
# Training a Non-Linear Model
#############################

model2 = NonLinearModule().to(device)
optimizer2 = optim.Adam(model2.parameters(), NONLINEAR_LEARNING_RATE)

# Put model in train mode
model2.train()
loss2 = []

for epoch in range(NONLINEAR_EPOCHS):
    # Batch training
    optimizer2.zero_grad()
    output = model2(X)
    loss = F.mse_loss(output, y)
    loss2.append(loss.detach().cpu().numpy())
    loss.backward()
    optimizer2.step()
    
    print(y[0], output[0])



#############################
# Training a Recurrent Model
#############################

# Fixing data

TIMESTEPS = 10
HIDDEN_SIZE = 64


rnn_data_x = []
rnn_data_y = []

for i in range(len(data1)-TIMESTEPS-1):
    rnn_data_x.append([np.concatenate((k[0], k[1])) for k in data1[i:i+TIMESTEPS]])
    rnn_data_y.append([k[2] for k in data1[i:i+TIMESTEPS]])

for i in range(len(data2)-TIMESTEPS-1):
    rnn_data_x.append([np.concatenate((k[0], k[1])) for k in data2[i:i+TIMESTEPS]])
    rnn_data_y.append([k[2] for k in data2[i:i+TIMESTEPS]])


rnn_data_x = torch.from_numpy(np.array(rnn_data_x)).type(torch.FloatTensor).to(device)
rnn_data_y = torch.from_numpy(np.array(rnn_data_y)).type(torch.FloatTensor).to(device)
# print("Y shape : ", rnn_data_y.shape)
loss3 = []
# hidden = torch.zeros((1, 1, HIDDEN_SIZE)).to(device)
model3 = RecurrentModule().to(device)
optimizer3 = optim.Adam(model3.parameters(), RECURRENT_LEARNING_RATE)

print(rnn_data_x.shape, rnn_data_y.shape)

for epochs in range(RECURRENT_EPOCHS):
    # Batch training
    # for i in range(len(rnn_data_x)):
    optimizer3.zero_grad()
    output, _ = model3(rnn_data_x)
    
    loss = F.mse_loss(output, rnn_data_y)
    loss3.append(loss.detach().cpu().numpy())
    loss.backward()
    optimizer3.step()

    print(loss.detach().cpu().numpy())
    # print(rnn_data_y[0], output[0])



#########################################
# Training a Hypernet Modulated Network
#########################################

# Transforming data

TIMESTEPS = 10
hypernet_data1 = []
hypernet_data2 = []

for i in range(len(data1)-TIMESTEPS-1):
    hypernet_data1.append(data1[i:i+TIMESTEPS])

for i in range(len(data2)-TIMESTEPS-1):
    hypernet_data2.append(data2[i:i+TIMESTEPS])

# hypernet_data1 = torch.tensor(hypernet_data1).to(device)
# hypernet_data2 = torch.tensor(hypernet_data2).to(device)
random.shuffle(hypernet_data1)
random.shuffle(hypernet_data2)

# Defining model and hyperparameters

LR1 = 0.01
LR2 = 0.05
HYPER_EPOCHS = 100

model = LearnableStory(device).to(device)

try:
    model.load_state_dict(torch.load("../saved_models/samplewise_inference.state"))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

hyper_optim = torch.optim.Adam(model.hypernet.parameters(), model.hyper_lr)
temporal_optim = torch.optim.Adam(model.temporal.parameters(), model.temporal_lr)
loss4 = []

for epochs in range(HYPER_EPOCHS):
    epoch_loss = []
    print("Epoch : ", epochs)
    
    for i in range(len(hypernet_data1)):
        hyper_optim.zero_grad()
        temporal_optim.zero_grad()
        
        # Ignore predicted and inferred states during training
        l1, _, _ = model(hypernet_data1[i]) 
        l2, _, _ = model(hypernet_data2[i])
        loss = (l1+l2)/2
        
        print("i = ", i)
        
        loss.backward()
        hyper_optim.step()
        temporal_optim.step()
        
        
        
        epoch_loss.append(loss.detach().cpu().numpy())
        
    loss4.append(np.mean(epoch_loss))

torch.save(model.state_dict(), "../saved_models/samplewise_inference.state")


#############################
# Plot figures
#############################
test_env1 = SimpleGridEnvironment(config=config1, goal=(2, 2), start_states=STARTS.copy())
test_env2 = SimpleGridEnvironment(config=config2, goal=(2, 2), start_states=STARTS.copy())
save_path = "../figures/dynamics_abstraction/"

plot_data1 = hypernet_data1[3]
plot_data2 = hypernet_data2[50]

print(plot_data1[0])
# model.eval()

loss, predicted_states, _ = model(plot_data1)
print("Predicted States : ", predicted_states, len(predicted_states))

for i in range(len(plot_data1)):
    start_state = plot_data1[i][0]
    start_state = start_state[2:]
    
    action = plot_data1[i][1]
    action = int(np.nonzero(action)[0][0])
    
    next_state = predicted_states[i][0]
    next_state = next_state[2:]
    print(f"Image {i} True vs Predicted: ", np.argmax(plot_data1[i][2][2:]), np.argmax(next_state))
    name = "transition_env_2_"+str(i)
    # print("Lengths : ", len(start_state), len(next_state))
    test_env1.plot_transition(start_state, action, next_state, save_path, name)
    

higher1_list = []
higher2_list = []

for i in range(len(hypernet_data1)):
    l1, _, higher1 = model(hypernet_data1[i]) 
    l2, _, higher2 = model(hypernet_data2[i])
    print(i, higher1)
    higher1_list.append(higher1)
    higher2_list.append(higher2)

h1 = np.array(higher1_list).astype('float32')
h2 = np.array(higher2_list).astype('float32')

h_ = np.concatenate([h1, h2], axis=0)
print("######### EMBEDDINGS #########")
print(h_)
print(h_.shape, type(h_))

tsne_ = TSNE(n_components=2, init='random', perplexity=5)
h_embed = tsne_.fit_transform(h_)
print(h_embed.shape)

# h_embed2 = TSNE(n_components=2, learning_rate='auto',\
#             init='random', perplexity=3).fit_transform(h2)

# x1 = [k[0] for k in higher1_list]
# y1 = [k[1] for k in higher1_list]

# x2 = [k[0] for k in higher2_list]
# y2 = [k[1] for k in higher2_list]
size = len(h_embed)
x1 = h_embed[:int(size/2), 0]
x2 = h_embed[int(size/2):, 0]

y1 = h_embed[:int(size/2), 1]
y2 = h_embed[int(size/2):, 1]

plt.clf()
plt.close()

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.legend()

plt.savefig("higher_latents.png")


#############################
# Plotting
#############################


plt.clf()
plt.close()

plt.plot(loss1, label="Linear")
plt.plot(loss2, label="Non-Linear")
plt.plot(loss3, label="Recurrent") 
plt.plot(loss4, label="Hypernet")

plt.legend()
plt.savefig("loss.png")




# model4 = HypernetMatrix(device).to(device)
# params1 = model4.hypernet.parameters()
# params2 = model4.temporal.parameters()

# opt1 = optim.Adam(params1, lr=LR1)  # Hypernet updates slowly
# opt2 = optim.SGD(params2, lr=LR2)   # Faster temporal updates

# print(params1, params2)

# # Training

# loss4 = []
# weights1, weights2 = [], []
# for epochs in range(HYPER_EPOCHS):
#     losses = []
    
#     for i in range(len(hypernet_data1)):
    
#         opt1.zero_grad()
#         opt2.zero_grad()
       
#         # Room 1 
#         higher_state = [0, 1]
#         loss1_ = model4(higher_state, hypernet_data1[i])
#         w1 = model4.weights_.detach().cpu().numpy()
        
#         # Room 2
#         higher_state = [1, 0]
#         loss2_ = model4(higher_state, hypernet_data2[i])
#         w2 = model4.weights_.detach().cpu().numpy()
        
#         # Updates
#         loss = loss1_+loss2_
#         loss.backward()
#         opt1.step()
#         opt2.step()
        
#         losses.append(loss.detach().cpu().numpy()/2)
    
#     print(f"#### EPOCH LOSS : {np.mean(losses)} ####")
#     print(f"Weights 1: {w1}")
#     print(f"Weights 2: {w2}")
#     loss4.append(np.mean(losses))

