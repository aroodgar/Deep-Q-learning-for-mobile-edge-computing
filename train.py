from fog_env import Offload
from RL_brain import DeepQNetwork
from plot import Plot
import numpy as np
import random
from utils import Utils
from data_utils import save_numpy_to_csv, convert_csv_to_excel
from gen_int_normal_dist import generate_episode_randint_normal
from config import Config
from QueueHandler import Queues
# import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):

    # still use reward, but use the negative value
    penalty = - max_delay * 2

    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay

    return reward


def train(results, iot_RL_list, NUM_EPISODE):

    RL_step = 0
    cummulative_drop_rate = 0.0
    sum_episode_drop_rate = 0.0
    for episode in range(NUM_EPISODE):

        print(episode)
        print(iot_RL_list[0].epsilon)
        # BITRATE ARRIVAL
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        bitarrive_p = np.random.randint(3, size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # =================================================================================================
        # ========================================= DRL ===================================================
        # =================================================================================================

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive, bitarrive_p)

        cnt = 0
        # TRAIN DRL
        while True:
            # PERFORM ACTION
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):

                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # should store this information in EACH time slot
            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index,:])

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for iot_index in range(env.n_iot):

                history[env.time_count - 1][iot_index]['observation'] = observation_all[iot_index, :]
                history[env.time_count - 1][iot_index]['lstm'] = np.squeeze(lstm_state_all[iot_index, :])
                history[env.time_count - 1][iot_index]['action'] = action_all[iot_index]
                history[env.time_count - 1][iot_index]['observation_'] = observation_all_[iot_index]
                history[env.time_count - 1][iot_index]['lstm_'] = np.squeeze(lstm_state_all_[iot_index,:])

                update_index = np.where((1 - reward_indicator[:,iot_index]) * process_delay[:,iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        iot_RL_list[iot_index].store_transition(history[time_index][iot_index]['observation'],
                                                                history[time_index][iot_index]['lstm'],
                                                                history[time_index][iot_index]['action'],
                                                                reward_fun(process_delay[time_index, iot_index],
                                                                           env.max_delay,
                                                                           unfinish_indi[time_index, iot_index]),
                                                                history[time_index][iot_index]['observation_'],
                                                                history[time_index][iot_index]['lstm_'])
                        iot_RL_list[iot_index].do_store_reward(episode, time_index,
                                                               reward_fun(process_delay[time_index, iot_index],
                                                                          env.max_delay,
                                                                          unfinish_indi[time_index, iot_index]))
                        iot_RL_list[iot_index].do_store_delay(episode, time_index,
                                                              process_delay[time_index, iot_index])
                        reward_indicator[time_index, iot_index] = 1

            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_
    
            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()
                    if iot_RL_list[iot].cost is not None:
                      cnt += 1
                      results[episode] += iot_RL_list[iot].cost
            # GAME ENDS
            if done:
                break
        print(f"task_arrive_prob = {env.task_arrive_prob}")
        print(f"trans_queue_p_dropped = {', '.join([str(i) for i in utils.dropped_trans_tasks_p])}")
        print(f"total trans tasks = {', '.join([str(i) for i in utils.total_trans_tasks_p])}")
        print(f"dropped and total fog tasks = {utils.dropped_fog_tasks}, {utils.total_fog_tasks}")
        print(f"dropped and total comp tasks = {utils.dropped_comp_tasks}, {utils.total_comp_tasks}")
        print(f"total dropped = {utils.dropped_tasks}")
        print(f"total avg proc delay = {utils.get_all_proc_delays_sum() / utils.done_tasks}")
        print(f"total tasks = {utils.total_tasks}")
        if cnt != 0:
          results[episode] /= cnt
          print(f'episode cost = {results[episode]}')   
        cumul_drop_rate = utils.dropped_tasks / utils.total_tasks
        cummulative_drop_rate += cumul_drop_rate
        if env.episode_task_count != 0:
          episode_drop_rate = (env.drop_trans_count + env.drop_iot_count + env.drop_fog_count) / env.episode_task_count
          sum_episode_drop_rate += episode_drop_rate
          print(f"episode_drop_rate = {episode_drop_rate}")
        print(f"cummulative avg drop rate = {cumul_drop_rate}")   
    print("RL_step =", RL_step)
    print(f"Final total drop rate = {cummulative_drop_rate / NUM_EPISODE}")
    print(f"episodes avg drop rate = {sum_episode_drop_rate / NUM_EPISODE}")
    print(f"comp tasks avg proc delay = {utils.sum_comp_proc_delay / utils.done_comp_tasks_count}")
    print(f"fog tasks avg proc delay = {utils.sum_fog_proc_delay / utils.done_fog_tasks_count}")
    print(f"total avg proc delay = {utils.get_all_proc_delays_sum() / utils.done_tasks}")
        #  =================================================================================================
        #  ======================================== DRL END=================================================
        #  =================================================================================================


if __name__ == "__main__":

    config = Config(QUEUE_TYPE=Queues.FIFO.name)

    NUM_IOT = config.num_iot
    NUM_FOG = config.num_fog
    NUM_EPISODE = config.num_episode
    NUM_TIME_BASE = config.num_time_base
    MAX_DELAY = config.max_delay
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY
    
    TASK_ARRIVE_PROB = config.task_arrive_prob

    QUEUE_TYPE = config.queue_type if not None else Queues.FIFO.name

    utils = Utils()

    # GENERATE ENVIRONMENT
    env = Offload(utils, NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY, QUEUE_TYPE, config.task_arrive_prob)

    # GENERATE MULTIPLE CLASSES FOR RL
    iot_RL_list = list()
    print("Testing")
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=config.lr,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=config.param_update_freq,  # each 200 steps, update target net
                                        memory_size=500,  # maximum of memory
                                        batch_size=config.batch_size
                                        ))

    # TRAIN THE SYSTEM
    results = [0] * NUM_EPISODE
    train(results, iot_RL_list, NUM_EPISODE)
    print('Training Finished')
    # plots = [{'x': range(10,NUM_EPISODE), 'y': results[10:]}]
    # names = ['gru tap=0.5']
    # p = Plot(plots, names)
    # p.draw_plot()
    save_numpy_to_csv(results, 'output')
    print('Saved outputs to csv.')
