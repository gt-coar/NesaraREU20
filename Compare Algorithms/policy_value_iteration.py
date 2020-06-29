#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:58:20 2020

@author: nesarasr
"""
import numpy as np
import time
import sys

# Policy Evaluation Function
def evaluate_policy(env, V, pi, gamma, theta):
    #print(V)
    #V = np.zeros((env.num_state,1))
    while True:
        delta = 0
        for s in env.states:
            v = V[s].copy()
            V=update_v_policy(env, V, pi, s, gamma)    #bellman update 
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


# Bellman Update function
def update_v_policy(env, V, pi, s, gamma):
    sum=0
    for a in env.actions:
        transitions = np.reshape(env.transitions[a][s][:],(-1,1))
        rewards = np.reshape(env.rewards[a][s][:],(-1,1))
        sum=sum+pi[s][a]*(np.sum(np.multiply(transitions,(rewards+(gamma*V)))))
    V[s]=sum
    return V

## Function that chooses the greedy action for a particular state 's'   
def choose_best_action_policy(env, V, pi, s, gamma):
    q=np.empty((env.num_action,1),dtype=float)
    for a in env.actions:
        pi[s][a]=0
        transitions = np.reshape(env.transitions[a][s][:],(-1,1))
        rewards = np.reshape(env.rewards[a][s][:],(-1,1))
        q[a]=np.sum(np.multiply(transitions,(rewards+(gamma*V))))
    action=np.argmax(q)        #Choose greedy action
    p=np.zeros(env.num_action)            #Update Policy
    p[action]=1
    return p


    
# Policy Improvement Function
def improve_policy(env, V, pi, gamma):
    policy_stable = True        # If policy_stable == True : Policy need not be updated anymore
    for s in env.states:
        old = pi[s].copy()
        pi[s]=choose_best_action_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old): 
            policy_stable = False
    return pi, policy_stable

# Policy Iteration
def policy_iteration(env, gamma, theta):
    V = np.zeros((env.num_state,1))          #Initialize Value function vector : [0,0,0...0]
    pi = np.ones((env.num_state,env.num_action)) / env.num_action   #Policy Initialization
    policy_stable = False
    i=0
    history=list()
    history_time=list()
    x=V.copy()
    history.append(x)
    history_time.append(0)
    start=time.time()
    while not policy_stable:
        i+=1
        V= evaluate_policy(env, V, pi, gamma, theta)          #Policy Evaluation step
        pi, policy_stable = improve_policy(env, V, pi, gamma)  #Policy Iteration step
        x=V.copy()
        history_time.append(time.time()-start)
        history.append(x)
    
    V= evaluate_policy(env, V, pi, gamma, theta) 
    x=V.copy()
    history_time.append(time.time()-start)
    history.append(x)
    print('Total number of iterations:',i)
    return V, pi,history,history_time





# Bellman greedy update
def update_v_value(env, V, s, gamma):
    q=np.empty((env.num_action,1),dtype=float)
    for a in env.actions:
        transitions = np.reshape(env.transitions[a][s][:],(-1,1))
        rewards = np.reshape(env.rewards[a][s][:],(-1,1))
        q[a]=np.sum(np.multiply(transitions,(rewards+(gamma*V))))
    action=np.argmax(q)
    return q[action]
    
    
## Function that chooses the greedy action for a particular state 's'   
def choose_best_action_value(env, V, pi, s, gamma):
    q=np.empty((env.num_action,1),dtype=float)
    for a in env.actions:
        pi[s][a]=0
        transitions = np.reshape(env.transitions[a][s][:],(-1,1))
        rewards = np.reshape(env.rewards[a][s][:],(-1,1))
        q[a]=np.sum(np.multiply(transitions,(rewards+(gamma*V))))
    action=np.argmax(q)        #Choose greedy action
    p=np.zeros(env.num_action)            #Update Policy
    p[action]=1
    return p
    
    
# Value Iteration
def value_iteration(env, gamma, theta):
    V = np.random.rand(env.num_state,1)
    i=0
    history=list()
    x=V.copy()
    history_time=list()
    history.append(x)
    history_time.append(0)
    start=time.time()
    while True:
        i+=1
        delta = 0
        for s in env.states:
            v = V[s].copy()
            V[s]=update_v_value(env, V, s, gamma)             #Bellman update
            delta = max(delta, abs(v - V[s]))
        x=V.copy()
        history_time.append(time.time()-start)
        history.append(x)
        if delta < theta:
            break
    pi = np.ones((env.num_state, env.num_action))/ env.num_action   #Initialize policy
    for s in env.states:
        pi[s]=choose_best_action_value(env, V, pi, s, gamma)    #Update policy
    V= evaluate_policy(env, V, pi, gamma, theta)
    x=V.copy()
    history_time.append(time.time()-start)
    history.append(x)
    print('Total number of iterations:',i)
    return V, pi,history,history_time