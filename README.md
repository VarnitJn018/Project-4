## **Project-4: Bayesian Optimisation for Hydrogen Production​**

The project employs a hybrid optimisation workflow for chemical experimentation. It uses Bayesian Optimisation (BoTorch/GPyTorch) to quantitatively recommend the next best experimental candidates, and a Large Language Model based on OpenAI GPT-4 to offer scientific reasoning and "sanity checks" on the suggested formulations. The system is intended to optimise a chemical mixture that includes ten distinct components in order to maximise hydrogen generation yield while adhering to rigorous chemical restrictions and discretisation requirements.

## **Bayesian Optimisation**

Imagine you have a function, let's call it $f(x)$, but you don't know the formula for it. This is a Black Box.

You can input a value $x$.
The box thinks for a while using an objective function (perhaps hours). 
It spits out a score $y$. 
Your goal is to find the $x$ that gives the highest $y$.

The Catch: Every time you try an $x$, it costs you something significant—time, money, or computational power. You cannot simply try every possible value or guess randomly forever. You need to be extremely smart about where you look next. 

Bayesian Optimisation solves this by building a probability model of the objective function and using it to decide where to sample next. It relies on two key components: 

  A.) The Surrogate Model (The Map): Since we don't know the true function $f(x)$, we build a probabilistic approximation of it, called a Surrogate model. The most common type used is a Gaussian Process (GP). 
          
  The GP does two things: Predicts the value of the function at points we haven't tested yet and quantifies the uncertainty of the surrogate model. 

  B.) The Acquisition Function (The Compass): Once we have our surrogate model (our "map"), we need a rule to decide where to sample next. This rule is the acquisition function. It looks at the surrogate model and calculates a score for every possible $x$, balancing two competing desires:
	        
  Exploitation: Looking in areas where the model predicts the value is high. 
          
  Exploration: Looking in areas where the uncertainty is high. 

## **The Bayesian Optimization Workflow**

  1. Initial Sampling: Evaluate the black box function at a few random points to get started.
	
  2. Fit Model: Fit the Surrogate Model (Gaussian Process) to the data you have observed so far.
	
  3. Optimise Acquisition: Use the Acquisition Function to find the next point $x_{new}$ that maximizes the trade-off between exploration and exploitation.
	
  4. Evaluate: Run the expensive Black Box function at $x_{new}$ to get the real result.
	
  5. Update: Add this new data point to your set and repeat from Step 2.
  
  6. You repeat this loop until you run out of time or budget.

<img style="text-align: center" width="700" height="789" alt="image" src="https://github.com/user-attachments/assets/28129640-cc6d-43d4-bcf4-2861c54c9384" />

## **Project-4 Details**

This project employs the following constraints in the BO model based on the data:
  
 1. Summation Constraints: The sum of all the components should not exceed 5.0
	
 2. Discrete steps: The quantity of each chemical should be a multiple of 0.25 gL
	
 3. Boundary condition: The value of P10-MIX1 can't fall below 1.2.

Following functions and models have been used in this BO workflow:
	
  a.) Surrogate model: [SingleTaskGP](https://botorch.org/docs/models/#single-task-gps)
  
  b.) Acquisition function: qLogNEI [(qLogNoisyExpectedImprovement)](https://botorch.org/docs/tutorials/closed_loop_botorch_only/)
	
  c.) Constraints: A custom function (apply_rounding_and_constraints)
	
  d.) Optimiser: [botorch.optim.optimize_acqf](https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf)
	
  e.) Likelihood: [gpytorch.mlls.ExactMarginalLogLikelihood](https://docs.gpytorch.ai/en/v1.9.0/_modules/gpytorch/mlls/exact_marginal_log_likelihood.html)

## **Pre-requisites**
  
  NumPy - Pandas - Matplotlib - scikit-learn - openai - botorch - gpytorch - API-keys for LLM models.

## **Authors**

Michael Huang, Joe Lowson, Guangrui Ai, Laurence Marlow, Varnit Jain, Emma Hall, and Dongyang Zheng​
