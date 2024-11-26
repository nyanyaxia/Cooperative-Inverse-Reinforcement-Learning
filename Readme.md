## Cooperative Inverse Reinforcement Learning

Implementation based on [this paper](https://papers.nips.cc/paper_files/paper/2016/hash/c3395dd46c34fa7fd8d729d8cf88b7a8-Abstract.html)

## Structure :

- run_test.py : running the experiment with best_response and expert_demonstration for 3 and 10 features
- gridworld.py : grid where robot and human move, with the different feature centers etc.
- robot.py : implement robot policy (learning through maximum entropy IRL : maxentwrapper.py and maxent_utils which comes from [here](https://pypi.org/project/irl-maxent/)
- human.py : demonstration trajectories
- best_response.py : to be implemented for human.py

## TODO :
- Fix best reponse policy to actually implement the algo
- Fix plots (need regret, KL divergence etc)