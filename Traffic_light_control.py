from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import traci
import random
import numpy as np
import keras
import h5py
from graph import Visualization
from DQNAgent import DQNAgent
from Sumo import Sumo
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

if __name__ == '__main__':
    sumoInt = Sumo()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    options = sumoInt.get_options()
    epsilondecay = 0.95
    rewards_all = []
    waiting_times_all = []
    queue_all = []

    path="D:/UM-Univerza v Mariboru/FERI/Seminarska/Models/"
    Visualization = Visualization(
        path,
        dpi=96
    )

    if options.nogui:
    #if True:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    vehNr = sumoInt.generate_sumo()



    episodes = 10
    batch_size = 32
    epsilon = 1


    agent = DQNAgent()


    try:
        agent.load('Models/model.h5')
    except:
        print('No model')

    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network

        #epsilon=0
        epsilon=1-(e / episodes)

        step = 0



        steps = 0
        action = 0

        reward1 = 0
        reward2 = 0
        sum_reward = 0
        total_reward = reward1 - reward2

        waiting_time = 0
        _waiting_times = {}
        total_waiting_time = 0
        sum_wait = 0

        traci.start([sumoBinary, "-c", "cross3ltl.sumocfg", '--start'])
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)
        print("--------------------")
        print("episode - " + str(e+1))


        while traci.simulation.getMinExpectedNumber() > 0 and steps < 7000:


            traci.simulationStep()
            state = sumoInt.getState()
            agent.epsilon = epsilon
            action = agent.act(state)
            #print(random.randrange(2))
            light = state[2]



            #incoming_roads = ["1si", "2si", "3si", "4si"]
            #car_list = traci.vehicle.getIDList()

            #for car_id in car_list:
            #    wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            #    road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            #    if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
            #        _waiting_times[car_id] = wait_time
            #        #print(wait_time)
            #    else:
            #        if car_id in _waiting_times: # a car that was tracked has cleared the intersection
            #            del _waiting_times[car_id]
            #total_waiting_time += sum(_waiting_times.values())



            if(action == 0 and light[0][0][0] == 0):
                # Transition Phase
                for i in range(6):
                    steps += 1
                    traci.trafficlight.setPhase('0', 1)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(10):
                    steps += 1
                    traci.trafficlight.setPhase('0', 2)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(6):
                    steps += 1
                    traci.trafficlight.setPhase('0', 3)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    steps += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            if(action == 0 and light[0][0][0] == 1):
                # Action Execution, no state change
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    steps += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            if(action == 1 and light[0][0][0] == 0):
                # Action Execution, no state change
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
                for i in range(10):
                    steps += 1
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
                    traci.trafficlight.setPhase('0', 0)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            if(action == 1 and light[0][0][0] == 1):
                for i in range(6):
                    steps += 1
                    traci.trafficlight.setPhase('0', 5)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(10):
                    steps += 1
                    traci.trafficlight.setPhase('0', 6)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(6):
                    steps += 1
                    traci.trafficlight.setPhase('0', 7)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
                for i in range(10):
                    steps += 1
                    traci.trafficlight.setPhase('0', 0)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()


            new_state = sumoInt.getState()
            reward = reward1 - reward2


            if reward < 0:
                sum_reward += reward

            sum_wait += reward2


            agent.remember(state, action, reward, new_state, False)
            # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
            if(len(agent.memory) > batch_size):
                agent.lrn(batch_size)



        #print("total waiting time",total_waiting_time,"\nsum waiting: ", sum_wait,"\n")

        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))

        print('Total waiting time - ' + str(sum_wait)+" epsilon - ", epsilon, " reward: "+str(sum_reward)+"queue length: "+str((waiting_time/steps)))
        #print("\nwaiting_time :",waiting_time)
        #print("\nsum_wait :",sum_wait)
        #print("\ntotal_waiting_time :",total_waiting_time)

        rewards_all.append(sum_reward)
        waiting_times_all.append(sum_wait)
        queue_all.append((waiting_time/steps)) #queue length


        #epsilon = epsilon * epsilondecay
        agent.save('reinf_traf_control_' + str(e) + '.h5')
        traci.close(wait=False)

        #waiting_time
        #reward
    Visualization.save_data_and_plot(data=rewards_all, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=waiting_times_all, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=queue_all, filename='queue', xlabel='Episode', ylabel='Cumulative queue length')




sys.stdout.flush()
