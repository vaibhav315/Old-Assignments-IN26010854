# ==========================================================
# CART POLE REINFORCEMENT LEARNING USING DEEP Q NETWORK
# ==========================================================


# Import Libraries


import gymnasium as gym

import numpy as np

import random

import matplotlib.pyplot as plt


from collections import deque



import tensorflow as tf

from tensorflow.keras import models, layers

from tensorflow.keras.optimizers import Adam




# ==========================================================
# 1. Create Environment
# ==========================================================


env = gym.make(
    "CartPole-v1"
)



state_size = env.observation_space.shape[0]

action_size = env.action_space.n



print("State Size:", state_size)

print("Action Size:", action_size)






# ==========================================================
# 2. Hyperparameters
# ==========================================================


episodes = 500


batch_size = 64


learning_rate = 0.001


gamma = 0.95


epsilon = 1.0


epsilon_min = 0.01


epsilon_decay = 0.995



memory = deque(
    maxlen=50000
)







# ==========================================================
# 3. Build Neural Network
# ==========================================================


def build_model():


    model = models.Sequential()


    model.add(

        layers.Dense(

            24,

            input_dim=state_size,

            activation="relu"

        )

    )



    model.add(

        layers.Dense(

            24,

            activation="relu"

        )

    )



    model.add(

        layers.Dense(

            action_size,

            activation="linear"

        )

    )



    model.compile(

        loss="mse",

        optimizer=Adam(

            learning_rate=learning_rate

        )

    )


    return model





model = build_model()


model.summary()






# ==========================================================
# 4. Store Experience
# ==========================================================



def remember(
    state,
    action,
    reward,
    next_state,
    done
):

    memory.append(

        (

        state,

        action,

        reward,

        next_state,

        done

        )

    )







# ==========================================================
# 5. Choose Action
# ==========================================================



def choose_action(state):


    global epsilon


    if np.random.rand() <= epsilon:

        return random.randrange(action_size)



    q_values = model.predict(

        state,

        verbose=0

    )


    return np.argmax(

        q_values[0]

    )







# ==========================================================
# 6. Train Agent
# ==========================================================



def replay():

    global epsilon


    if len(memory) < batch_size:

        return



    batch=random.sample(

        memory,

        batch_size

    )



    for state,action,reward,next_state,done in batch:


        target = reward



        if not done:


            target = reward + gamma * np.max(

                model.predict(

                    next_state,

                    verbose=0

                )[0]

            )



        target_f = model.predict(

            state,

            verbose=0

        )


        target_f[0][action]=target



        model.fit(

            state,

            target_f,

            epochs=1,

            verbose=0

        )



    if epsilon > epsilon_min:


        epsilon *= epsilon_decay







# ==========================================================
# 7. Training Loop
# ==========================================================



scores=[]



for episode in range(episodes):


    state,info = env.reset()



    state=np.reshape(

        state,

        [1,state_size]

    )


    total_reward=0



    done=False



    while not done:


        action=choose_action(

            state

        )



        next_state,reward,terminated,truncated,info = env.step(

            action

        )



        done = terminated or truncated



        next_state=np.reshape(

            next_state,

            [1,state_size]

        )



        remember(

            state,

            action,

            reward,

            next_state,

            done

        )



        state=next_state


        total_reward+=reward



        if done:

            break




    replay()



    scores.append(

        total_reward

    )



    print(

        "Episode:",

        episode+1,

        "Score:",

        total_reward,

        "Epsilon:",

        epsilon

    )







# ==========================================================
# 8. Reward Graph
# ==========================================================



plt.figure(figsize=(10,5))


plt.plot(

scores

)


plt.xlabel(

"Episode"

)


plt.ylabel(

"Reward"

)


plt.title(

"CartPole Training Performance"

)


plt.show()







# ==========================================================
# 9. Test Trained Agent
# ==========================================================



test_env = gym.make(

    "CartPole-v1",

    render_mode="human"

)



state,info=test_env.reset()


state=np.reshape(

    state,

    [1,state_size]

)



total_reward=0



done=False



while not done:


    action=np.argmax(

        model.predict(

            state,

            verbose=0

        )

    )



    next_state,reward,terminated,truncated,info=test_env.step(

        action

    )



    done=terminated or truncated



    state=np.reshape(

        next_state,

        [1,state_size]

    )


    total_reward+=reward



print(

"Test Score:",

total_reward

)



test_env.close()







# ==========================================================
# 10. Save Model
# ==========================================================



model.save(

"cartpole_dqn_model.h5"

)



print(

"Model Saved Successfully!"

)
