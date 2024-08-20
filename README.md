# Navigating Trade-offs: Policy Summarization for Multi-Objective Reinforcement Learning
Official code repository for ["Navigating Trade-offs: Policy Summarization for Multi-Objective Reinforcement Learning (ECAI-2024)]().  


*Multi-Objective Reinforcement Learning outputs a set of policies, where each offers different trade-offs concerning discounted rewards, as can be seen on the plot below for the Highway environment. 
Each policy is represented as a dot, and its discounted returns are on the X and Y axes, where there are trade-offs between staying on the right lane and obtaining high speed.* 

<img src="solutions_highway.png" alt="Objectives" width="900"/> 

*With policy summarization, the trade-off of the policies can also be represented by the way each policy behaves using policy summarization techniques. In our paper, we used
[Highlights, AAMAS 2018](https://scholar.harvard.edu/files/oamir/files/highlightsmain.pdf) to highlight the most important trajectories for each policy.*

<img src="P1_Highlights.gif" alt="P1" width="900"/> 

<img src="P5_Highlights.gif" alt="P5" width="900"/> 





### Installation  
  
The project is based on Python 3.7. All the necessary packages are in requirements.txt.
Create a virtual environment and install the requirements using:
```
pip install -r requirements.txt
```

### Required repositories
The Highway domain implementation of the algorithm requires the following repositories:

[https://github.com/eleurent/highway-env](https://github.com/eleurent/highway-env) V1.4

[https://github.com/eleurent/rl-agents](https://github.com/eleurent/rl-agents)

### Adding a new domain
The code works by accessing an **interface** for each domain.
Adding a new domain requires a relevant interface and configuration file to be added to *disagreements/Interfaces* and *disagreements/configs* respectively.

### Running
The *configuration_dict* dictionary in *run_comparison.py* should be updated for any new agents or domains you wish to compare. 
```
python run_comparison.py -a1 ClearLane -a2 FastRight
```
