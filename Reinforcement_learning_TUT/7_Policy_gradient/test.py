import numpy as np
# import cPickle as pickle
# import gym
#
# # hyperparameters
# H = 200  # number of hidden layer neurons
# batch_size = 10  # every how many episodes to do a param update?
# learning_rate = 1e-4
# gamma = 0.99  # discount factor for reward
# decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
# resume = False  # resume from previous checkpoint?
# render = False
#
# # model initialization
# D = 80 * 80  # input dimensionality: 80x80 grid
# if resume:
#     model = pickle.load(open('save.p', 'rb'))
# else:
#     model = {}
#     model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
#     model['W2'] = np.random.randn(H) / np.sqrt(H)
#
# grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
# rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory
#
#
# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]
#
#
# def prepro(I):
#     """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#     I = I[35:195]  # crop
#     I = I[::2, ::2, 0]  # downsample by factor of 2
#     I[I == 144] = 0  # erase background (background type 1)
#     I[I == 109] = 0  # erase background (background type 2)
#     I[I != 0] = 1  # everything else (paddles, ball) just set to 1
#     return I.astype(np.float).ravel()
#
#
# def discount_rewards(r):
#     """ take 1D float array of rewards and compute discounted reward """
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(xrange(0, r.size)):
#         if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r
#
#
# def policy_forward(x):
#     h = np.dot(model['W1'], x)
#     h[h < 0] = 0  # ReLU nonlinearity
#     logp = np.dot(model['W2'], h)
#     p = sigmoid(logp)
#     return p, h  # return probability of taking action 2, and hidden state
#
#
# def policy_backward(eph, epdlogp):
#     """ backward pass. (eph is array of intermediate hidden states) """
#     dW2 = np.dot(eph.T, epdlogp).ravel()
#     dh = np.outer(epdlogp, model['W2'])
#     dh[eph <= 0] = 0  # backpro prelu
#     dW1 = np.dot(dh.T, epx)
#     return {'W1': dW1, 'W2': dW2}
#
#
# env = gym.make("Pong-v0")
# observation = env.reset()
# prev_x = None  # used in computing the difference frame
# xs, hs, dlogps, drs = [], [], [], []
# running_reward = None
# reward_sum = 0
# episode_number = 0
# while True:
#     if render: env.render()
#
#     # preprocess the observation, set input to network to be difference image
#     cur_x = prepro(observation)
#     x = cur_x - prev_x if prev_x is not None else np.zeros(D)
#     prev_x = cur_x
#
#     # forward the policy network and sample an action from the returned probability
#     aprob, h = policy_forward(x)
#     action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
#
#     # record various intermediates (needed later for backprop)
#     xs.append(x)  # observation
#     hs.append(h)  # hidden state
#     y = 1 if action == 2 else 0  # a "fake label"
#     dlogps.append(
#         y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
#
#     # step the environment and get new measurements
#     observation, reward, done, info = env.step(action)
#     reward_sum += reward
#
#     drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
#
#     if done:  # an episode finished
#         episode_number += 1
#
#         # stack together all inputs, hidden states, action gradients, and rewards for this episode
#         epx = np.vstack(xs)
#         eph = np.vstack(hs)
#         epdlogp = np.vstack(dlogps)
#         epr = np.vstack(drs)
#         xs, hs, dlogps, drs = [], [], [], []  # reset array memory
#
#         # compute the discounted reward backwards through time
#         discounted_epr = discount_rewards(epr)
#         # standardize the rewards to be unit normal (helps control the gradient estimator variance)
#         discounted_epr -= np.mean(discounted_epr)
#         discounted_epr /= np.std(discounted_epr)
#
#         epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
#         grad = policy_backward(eph, epdlogp)
#         for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch
#
#         # perform rmsprop parameter update every batch_size episodes
#         if episode_number % batch_size == 0:
#             for k, v in model.iteritems():
#                 g = grad_buffer[k]  # gradient
#                 rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
#                 model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
#                 grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
#
#         # boring book-keeping
#         running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
#         print
#         'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
#         if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
#         reward_sum = 0
#         observation = env.reset()  # reset env
#         prev_x = None
#
#     if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
#         print('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')







import gym
import tensorflow as tf
env = gym.make('CartPole-v0')

# hyperparameters
H = 10  # number of hidden layer neurons
batch_size = 50  # every how many episodes to do a param update?
learning_rate = 2e-2  # feel free to play with this to train faster or more stably.
gamma = 0.99  # discount factor for reward

D = 4  # input dimensionality

tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to
#giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)  # prob for action=0, label=1

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = input_y*tf.log(probability) + (1 - input_y)*tf.log(1-probability)     # p=1, a=0, y=1; p=0, a=1, y=0
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()  # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 0 if np.random.uniform() < tfprob else 1     # p=1, a=0; and p=0, a=1

        xs.append(x)  # observation
        y = 1 if action == 0 else 0  # a "fake label", p=1, a=0, y=1; p=0, a=1, y=0
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)

        # x, x_dot, theta, theta_dot = observation
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 2
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians
        # reward = r1 + r2

        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad, tloss = sess.run([newGrads, loss], feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            print(tloss)
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(
                'Average reward for episode %f.  Total average reward %f.' % (
                reward_sum / batch_size, running_reward / batch_size))

                if reward_sum / batch_size > 200:
                    print(
                    "Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(
episode_number, 'Episodes completed.')