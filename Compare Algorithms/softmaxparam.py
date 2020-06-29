#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:27:55 2020

@author: nesarasr
"""

from policyiteration import evaluate_policy
import numpy as np
import time


def softmax(env,pi_theta):
    pi = np.zeros((env.num_state,env.num_action),dtype=float)
    for s in env.states:
        pi[s]=np.exp(pi_theta[s])
        pi[s]=pi[s]/np.sum(pi[s])
    return np.array(pi)

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

def grad(env,gamma,rho,pi_theta,theta):
    V_grad = np.zeros((env.num_state,env.num_action))
    pi = softmax(env,pi_theta)
    #print(pi_theta)
    P = P_matrix(env,pi)
    V = evaluate_policy(env, pi, gamma, theta)
    #print(V_grad[0][1])
    for s in range(0,env.num_state):
        d=d_pi_s(env,P,gamma,s)
        d_rho= np.dot(rho,d)
        for a in range(0,env.num_action):
            transitions = np.reshape(env.transitions[a][s][:],(-1,1))
            rewards = np.reshape(env.rewards[a][s][:],(-1,1))
            q=np.sum(np.multiply(transitions,rewards)+ gamma * np.multiply(transitions,V))
            advantage=q-V[s]
            V_grad[s][a]= (d_rho*advantage*pi[s][a])/(1-gamma)
    
    
    return V_grad
            
def policy_gradient_constant_step(env,gamma,theta,rho,iterations,learning_rate):
    pi_theta = np.ones((env.num_state,env.num_action)) / env.num_action   #Policy Initialization
    V = np.zeros((env.num_state,1))          #Initialize Value function vector : [0,0,0...0]
    # min_list=list()
    # min_t = (V_rho_optimal-np.dot(rho,V)[0])
    # min_list.append(min_t)
    
    history=list()
    history_time=list()
    x=V.copy()
    history.append(x)
    history_time.append(0)
    start=time.time()
    
    for i in range(0,iterations):
        V_grad=grad(env,gamma,rho,pi_theta,theta)
        for s in env.states:
            pi_theta[s]= pi_theta[s]+ learning_rate * V_grad[s]
        pi = softmax(env,pi_theta)
        V=evaluate_policy(env, pi, gamma, theta) 
        x=V.copy()
        history_time.append(time.time()-start)
        history.append(x)
        # min_t=min(min_t,(V_rho_optimal-np.dot(rho,V)[0]))
        # min_list.append(min_t)
    return pi,V,history,history_time      #min_list
        

def policy_gradient_variant_step(env,gamma,theta,rho,iterations,alpha):
    pi_theta = np.ones((env.num_state,env.num_action)) / env.num_action   #Policy Initialization
    V = np.zeros((env.num_state,1))          #Initialize Value function vector : [0,0,0...0]
    # min_list=list()
    # min_t = (V_rho_optimal-np.dot(rho,V)[0])
    # min_list.append(min_t)
    
    history=list()
    history_time=list()
    x=V.copy()
    history.append(x)
    history_time.append(0)
    start=time.time()
    learning_rate = 1
    for i in range(0,iterations):
        if i==0:
            pass
        elif i%10==0:
            learning_rate= 1/(np.power(i,alpha))
        V_grad=grad(env,gamma,rho,pi_theta,theta)
        for s in env.states:
            pi_theta[s]= pi_theta[s]+ learning_rate * V_grad[s]
        pi = softmax(env,pi_theta)
        V=evaluate_policy(env, pi, gamma, theta) 
        x=V.copy()
        history_time.append(time.time()-start)
        history.append(x)
        # min_t=min(min_t,(V_rho_optimal-np.dot(rho,V)[0]))
        # min_list.append(min_t)
    return pi,V,history,history_time       #min_list
        