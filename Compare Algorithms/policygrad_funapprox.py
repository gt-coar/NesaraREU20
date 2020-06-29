#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 01:55:38 2020

@author: nesarasr
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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


def softmax(env,theta,phi):
    pi = np.zeros((env.num_state,env.num_action),dtype=float)
    for s in env.states:
        for a in env.actions:
            pi[s][a]=np.exp(np.dot(np.array(phi[s][a]),theta))
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

def evaluate_q(env, pi, gamma, V):
    Q = np.zeros((env.num_state,env.num_action))
    for s in env.states:
        for a in env.actions:
            transitions = np.reshape(env.transitions[a][s][:],(-1,1))
            rewards = np.reshape(env.rewards[a][s][:],(-1,1))
            Q[s][a] = np.sum(np.multiply(transitions,rewards)+ gamma * np.multiply(transitions,V))
    return Q



def phi_tilde(env,pi,phi,s,d):
    phi_s=np.zeros(d)  #np.zeros replacee
    for a in env.actions:
        phi_s = phi_s+phi[s][a]*pi[s][a]
    #print(phi_s)
    return phi_s
    
def policygradient_funapprox(env,gamma,phi,step_size,iterations,d,EPS,rho,V_rho_optimal):
    theta = np.zeros((d,1))
    w  = np.zeros((d,1))
    
    for i in range(0,iterations):
        pi = softmax(env,theta,phi)
        #print(pi)
        P = P_matrix(env,pi)
        V = evaluate_policy(env, pi, gamma, EPS)
        Q_pi = evaluate_q(env, pi, gamma, V)
        d_rho = np.zeros((env.num_state,1))
        for s in env.states:
            d_s = d_pi_s(env,P,gamma,s)
            d_rho = d_rho+ rho[s]*d_s 
    
        b= np.zeros((d,1))
        A= np.zeros((d,d))
        theta_u = np.zeros((d,1))
        for s in env.states:
            phi_tilde_s = phi_tilde(env,pi,phi,s,d)
            for a in env.actions:
                phi_t = (phi[s][a]-phi_tilde_s)
                #print(np.dot(phi_t,np.transpose(w)[0]))
                dfw = phi_t * pi[s][a]
                theta_u = theta_u + d_rho[s]* np.reshape(dfw,(d,1)) * np.dot(phi_t,np.transpose(w)[0])
                b = b + d_rho[s]* np.reshape(dfw,(d,1)) * Q_pi[s][a]
                x = d_rho[s] * dfw
                for j in range(0,d):
                    #print(j)
                    A[j] = A[j] + x*phi_t[j]
        #print(A)
        #print(b)
        w = np.dot(np.linalg.inv(A),b)
        #print(w)
        theta = theta +  step_size*theta_u
        #print(theta)
    return theta
   
    