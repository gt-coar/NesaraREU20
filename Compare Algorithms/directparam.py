#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:27:25 2020

@author: nesarasr
"""

import numpy as np
import sys
import time
import warnings
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import os


class HiddenPrints:       #To suppress print statements during execution of library function
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def policy(s,a):
    pi = np.ones((s,a)) / a   #Policy Initialization
    return pi   


def P_matrix(env,pi):
    
    # Compute Transition probabilities P(s,s') using P(s'| s,a ) and pi( a|s ) : Matrix of size |s| x |s|
    P=np.empty((env.num_state,env.num_state),dtype=float)
    P=np.multiply(np.reshape(pi[:,0],(-1,1)),env.transitions[0][:][:])
    for a in range(1,env.num_action):
        P=P+np.multiply(np.reshape(pi[:,a],(-1,1)),env.transitions[a][:][:])
    return P
        
def d_pi_s(env,P,gamma,s):
    e_s=np.zeros((env.num_state,1))
    e_s[s]=1
    i= np.identity(env.num_state)- gamma*(P)
    d= (1-gamma) * np.dot(np.linalg.inv(i),e_s)
    return d



def projection(env,x):
    P = 2*matrix(np.identity(env.num_action))
    q = -2*matrix(x)
    G = -1*matrix(np.identity(env.num_action))
    h = matrix(np.zeros(env.num_action))
    A = matrix(np.ones(env.num_action), (1,env.num_action))
    b = matrix(1.0)
    with HiddenPrints():
        sol=solvers.qp(P, q, G, h, A, b)
    proj= np.reshape(np.array(sol['x']),(1,-1))
    return proj

# Policy Evaluation Function
def evaluate_policy(env, pi, gamma, theta):
    V = np.zeros((env.num_state,1))
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




def grad(env,gamma,rho,pi,theta):
    V_grad= np.empty((env.num_state,env.num_action),dtype=float)
    P=P_matrix(env,pi)
    V=evaluate_policy(env, pi, gamma, theta)
    for s in env.states:
        d=d_pi_s(env,P,gamma,s)
        d_rho= np.dot(rho,d)
        for a in env.actions:
            transitions = np.reshape(env.transitions[a][s][:],(-1,1))
            rewards = np.reshape(env.rewards[a][s][:],(-1,1))
            q=np.sum(np.multiply(transitions,rewards)+ gamma * np.multiply(transitions,V))
            V_grad[s][a]= (d_rho*q)/(1-gamma)
    
    return V_grad
            
        
def policy_gradient(env,gamma,theta,rho,iterations,learning_rate):
    pi = np.ones((env.num_state,env.num_action)) / env.num_action   #Policy Initialization
    V = np.zeros((env.num_state,1))          #Initialize Value function vector : [0,0,0...0]
    #min_list=list()
    #min_t = (V_rho_optimal-np.dot(rho,V)[0])
    #min_list.append(min_t)
    history=list()
    history_time=list()
    x=V.copy()
    history.append(x)
    history_time.append(0)
    start=time.time()
    for i in range(0,iterations):
        V_grad=grad(env,gamma,rho,pi,theta)
        for s in env.states:
            pi_update= pi[s]+ learning_rate * V_grad[s]
            pi[s]= projection(env,pi_update) 
        V=evaluate_policy(env, pi, gamma, theta) 
        x=V.copy()
        history_time.append(time.time()-start)
        history.append(x)
       # min_t=min(min_t,(V_rho_optimal-np.dot(rho,V)[0]))
       # min_list.append(min_t)
    return pi,V, history,history_time       #min_list
        