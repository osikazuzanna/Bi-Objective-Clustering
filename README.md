# Navigating Trade-offs: Policy Summarization for Multi-Objective Reinforcement Learning
Official code repository for ["Navigating Trade-offs: Policy Summarization for Multi-Objective Reinforcement Learning (ECAI-2024)]().  


*Multi-Objective Reinforcement Learning outputs a set of policies, where each offers different trade-offs concerning discounted rewards, as can be seen on the plot below for the Highway environment. 
Each policy is represented as a dot, and its discounted returns are on the X and Y axes, where there are trade-offs between staying on the right lane and obtaining high speed.* 

<img src="solutions_highway.png" alt="Objectives" width="900"/> 

*With policy summarization, the trade-off of the policies can also be represented by the way each policy behaves using policy summarization techniques. In our paper, we used
[Highlights, AAMAS 2018](https://scholar.harvard.edu/files/oamir/files/highlightsmain.pdf) to highlight the most important trajectories for each policy.*



<img src="P1_Highlights.gif" alt="P1" width="900"/> 

<img src="P5_Highlights.gif" alt="P5" width="900"/> 


*To show that similar trade-offs in the objective space doesn't necessary mean similar behaviour, we show highlights for 2 policies, Policy 1 (P1) and Policy 5 (P5), which are similar to each other based on the scatter plot in the objective space. However, when you study the behaviour based on the highlights, you can notice that the style of taking over other cars differs between two policies - see Highlight 1 for P1 and Highlight 2 for P5. These differences are crucial in decision-making scenarios, where a decision maker has to choose one policy from a set of solutions.*

**Our paper shows how this can be used in a decision-making scenario to support a DM in making a decision in a complex, multi-dimensional set of solutions**

### Installation  
  
The project is based on Python 3.9. All the necessary packages are in requirements.txt.
Create a virtual environment and install the requirements using:
```
pip install -r requirements.txt
```

## Requierements

### Required repositories
The implementation of MORL agents requires the following repository:

[https://github.com/LucasAlegre/morl-baselines](https://github.com/LucasAlegre/morl-baselines)

### Required Inputs
The code is divided into two parts:
  * *generating_input*, which generates distance matrices in both spaces (behavior and objective), given: a trained agent, the environment, the pareto-weights and the objective values
  * *src*, with source code for PAN, which is used to compute the clusters using bi-objective clustering approach

