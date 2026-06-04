"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA =  0.1 #1 #10**(-6)  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    # TODO Your code here

    # generate random number between 0 and 1 (r)
    rng = np.random.default_rng()

    r = rng.random()
    #print(r)

    #if r < epsilon (epsilon is the treshold)
        #explore - choose random action index and random object index
    if r < epsilon:
        action_index = rng.integers(NUM_ACTIONS)
        object_index = rng.integers(NUM_OBJECTS)

    # else exploit 
        #look at all Q values for the current state
        # find best action-object value and use that pair
    else:
        current_state = q_func[state_1, state_2]
        best_pair = np.argmax(current_state)
        
        action_index, object_index = np.unravel_index(best_pair, current_state.shape)

    #action_index, object_index = None, None
    return (action_index, object_index)

#testing epsilon greedy for epsilon = 0 (testing the exploit phase)
# q_test = np.zeros((2, 2, 3, 4))
# q_test[0, 0, 1, 2] = 100 # to make this obviously the best to test if it will choose correctly
# action_i, object_i = epsilon_greedy(state_1=0,state_2=0,q_func=q_test,epsilon=0)
# print(action_i, object_i) #should print 1 2 (passed test)

#testing epsilon greedy for epsilon = 0 (testing the exploration phase)
# q_test = np.zeros((2, 2, 3, 4))
# q_test[0, 0, 1, 2] = 100

# for i in range(20):
#     action_i, object_i = epsilon_greedy(state_1=0,state_2=0,q_func=q_test,epsilon=1)
#     print(action_i, object_i) #should print a different answer for every loop and return random pairs
# passed test!


# # pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    # TODO Your code here
    
    # get the current q value
    current_q = q_func[current_state_1, current_state_2, action_index, object_index]

    #if it is terminal state
        #target = reward
    if terminal == True:
        target = reward
    
    #else
        # look at all q values for next state
        #find max q value for next state
        # target = reward + gamma * max of next state
    else:
        next_q = q_func[next_state_1, next_state_2]
        max_next_q = np.max(next_q)
        target = reward + GAMMA * max_next_q

    # new q = current q + alpha* (target - current q)
    # updtae q-table

    new_q = current_q + ALPHA * (target - current_q)

    q_func[current_state_1, current_state_2, action_index,
           object_index] = new_q  # TODO Your update here

    return None  # This function shouldn't return anything

# #testing tabular q learning function for terminal = False
# q_test = np.zeros((2, 2, 3, 4))
# current_state_1 = 0; current_state_2 = 0; action_index = 1; object_index = 2
# q_test[0,0,1,2] = 5 #current q value
# q_test[1,1,0,0] = 10 # next state best q value

# tabular_q_learning(q_test, current_state_1=0, current_state_2=0, action_index=1, object_index=2, reward=2, next_state_1=1, next_state_2=1, terminal = False)
# print(q_test[0,0,1,2]) # should print 5.2 (passed the test!!!)

# # Testing tabular q learning function for terminal = True
# q_test = np.zeros((2,2,3,4))
# q_test[0,0,1,2] = 5
# tabular_q_learning(q_test, current_state_1=0, current_state_2=0, action_index=1, object_index=2, reward=2, next_state_1=1, next_state_2=1, terminal = True)
# print(q_test[0,0,1,2]) # should print 4.7 (passed test!!!)

# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = None
    # initialize for each episode
    # TODO Your code here
    #If testing, epi = 0, t = 0
    epi_reward = 0
    step = 0

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    # no TODO here but we need to add these lines to convert the descriptions into room and quest indexes
    current_state_1 = dict_room_desc[current_room_desc]
    current_state_2 = dict_quest_desc[current_quest_desc]

    while not terminal:
        # Choose next action and execute
        # TODO Your code here
        # Choose action - USE epsilon greedy to choose action
        action_index, object_index = epsilon_greedy(current_state_1, current_state_2, q_func, epsilon)
        # Execute action - convert action and object indexes into strings
        # Update environment and state based on action
        act = ACTIONS[action_index]
        obj = OBJECTS[object_index]
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(current_room_desc, current_quest_desc, action_index, object_index)        
        # Check reward, next room, next quest, terminal
        # Convert next descriptions into next indices
        next_state_1 = dict_room_desc[next_room_desc]
        next_state_2 = dict_quest_desc[next_quest_desc]

        if for_training:
            # update Q-function.
            # TODO Your code here
            # run tabular_q_learning
            tabular_q_learning(q_func, current_state_1, current_state_2, action_index, object_index, reward, next_state_1, next_state_2, terminal)
            pass

        if not for_training:
            # update reward
            # TODO Your code here
            # update epi_reward <- epi_reward + gamma^step * reward
            # update step
            epi_reward = epi_reward + (GAMMA ** step) * reward
            step = step + 1
            pass

        # prepare next step
        # TODO Your code here
        # move to next step by updating current state <- next state
        current_state_1 = next_state_1
        current_state_2 = next_state_2
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return epi_reward


# pragma: coderesponse end


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()

print('\n Question 2: What is the number of epochs when the learning algorithm converges?\n')
print ('Answer: The learning algorithm converges at around the 15th epoch. Looking at the figure, the testing reward grows quickly until around 15 epochs, and then stabilizes with minor fluctuations\n')

print('\n Question 3: What is the average episodic rewards of your Q-learning algorithm when it converges?\n')
print ('Answer: The average episodic reward when it converges is around 0.52, with slight fluctuations from 0.50 to 0.53. This value was taken from the plot. We do not choose the Avg reward printed, as that averages across the entire learning process, including the bad initial epochs. \n')

print('\n Question 4: Which of the behaviors below do you observe from running the algorithm?\n')
print('Answer: For very large ϵ (say ϵ = 1) the algorithm converges slower compared to ϵ = 0.5. (Using ϵ = 1, took 30 epochs to converge ).\n For very small ϵ (say ϵ = 0.000001) the algorithm actually converges at around the same value as for ϵ = 0.5 (around 15 epochs).\n')

print ('\n Question 5: Fix the exploration parameter ϵ = 0.5 and do the experiments with different values of the training alpha ∈ [10-6, 1]. What do you observe?\n')
print ('Answer: The algorithm converges for all values of alpha in less than 200 epochs (using the definition of convergence to be that the testing performance becomes stable). We observe convergence for all alpha even though it may converge to lower performance within the fixed training horizon.\n In terms of the speed of convergence, alpha does not seem to have any effect on it, as for both very low and very high alpha values, the convergence seems to be reached at around 15 epochs.')