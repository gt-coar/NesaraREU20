# NesaraREU20
### Nesara's REU 2020 Project Code Repository

#### Contents

* __Plot J__
  * Contains the implementation of 3D visualization of the objective function of a 2 State 2 Action MDP
* __Policy and Value Iteration__ 
  * Policy Iteration 
  * Value Iteration
  * Policy vs Value Iteration (Comparison) </br>
* __Direct Parameterization__
  * Contains the implementation of Direct Policy Parameterization </br>
* __SoftMax Parameterization__
  * Contains the implementation of SoftMax Policy Parameterization </br>
* __Natural actor critic__
  * Contains the implementation of the paper : <b><i>"Finite Sample Analysis of Two-Time-Scale NaturalActor-Critic Algorithm"</i></b>
* __Natural Policy Gradient : MDP Case__
  * Natural Policy Gradient with Softmax Parameterization
  * Natural Policy Gradient with Softmax Parameterization : Function Approximation 
* __Mirror and Lazy Mirror Descent__
  * Mirror Descent
  * Lazy Mirror Descent 
* __TRPO__ 
  * Contains the implementation of Trust Region Policy Optimization Algorithm
* __Compare Algorithms__ : Notebooks containing comparisons between various algorithms implemented in this repository
  * Compare time and iteration: Compares the total time and total iterations taken by the following algorithms to converge in separate plots
    * Policy Iteration
    * Value Iteration
    * Policy Gradient with Direct Parameterization (Constant step-size)
    * Policy Gradient with Direct Parameterization (Time variant step-size)
    * Policy Gradient with Softmax Parameterization (Constant step-size)
    * Policy Gradient with Softmax Parameterization (Time variant step-size)
  * Compare mirror and lazy mirror descent : Compares the convergence between mirror and lazy mirror descent algorithms
  * Compare convergence of policy gradient algos : Compares the total iterations taken by the following policy gradient based algorithms to converge 
    * Policy Gradient with Direct Parameterization (Constant step-size)
    * Policy Gradient with Softmax Parameterization (Constant step-size)
    * Policy Gradient with Softmax Parameterization (Constant step-size) : Function Approximation
    * Natural Policy Gradient with Softmax Parameterization (Constant step-size)
    * Natural Policy Gradient with Softmax Parameterization (Constant step-size) : Function Approximation
