---
layout: post
title: Intro to Thompson Sampling Part 1: the Bernoulli Bandit
featured-img: ts_for_mab_cover
mathjax: true
---

# Interactive intro to Thompson Sampling Part 1: the Bernoulli Bandit

Thompson Sampling is a very simple yet effective method to addressing the exploration-exploitation dilemma in reinforcement/online learning. In this series of posts, I'll introduce some applications of Thompson Sampling in simple examples, trying to show some cool visuals along the way. All the code can be found on my GitHub page [here](https://github.com/gdmarmerola/interactive-intro-rl).

In this post, we explore the simplest setting of online learning: the Bernoulli bandit.

## Problem: The Bernoulli Bandit

The  Multi-Armed Bandit problem is the simplest setting of reinforcement learning. Suppose that a gambler faces a row of slot machines (bandits) on a casino. Each one of the $K$ machines has a probability $\theta_k$ of providing a reward to the player. Thus, the player has to decide which machines to play, how many times to play each machine and in which order to play them, in order to maximize his long-term cumulative reward. 

<img src = "https://github.com/gdmarmerola/gdmarmerola.github.io/blob/master/assets/img/ts_for_mab/multiarmedbandit.jpg">

At each round, we receive a binary reward, taken from an Bernoulli experiment with parameter $\theta_k$. Thus, at each round, each bandit behaves like a random variable $Y_k \sim \textrm{Binomial}(\theta_k)$. This version of the Multi-Armed Bandit is also called the Binomial bandit.

We can easily define in Python a set of bandits with known reward probabilities and implement methods for drawing rewards for them. We also compute the **regret**, which is the difference $\theta_{best} - \theta_i$ of the expected reward $\theta_i$ of our chosen bandit $i$ and the largest expected reward $\theta_{best}$.

```python
# class for our row of bandits
class MAB:
    
    # initialization
    def __init__(self, bandit_probs):
        
        # storing bandit probs
        self.bandit_probs = bandit_probs
        
    # function that helps us draw from the bandits
    def draw(self, k):

        # we return the reward and the regret of the action
        return np.random.binomial(1, self.bandit_probs[k]), np.max(self.bandit_probs) - self.bandit_probs[k]
```

We also can use **matplotlib** to generate a video of random draws from these bandits. Each row shows us the history of draws for the corresponding bandit, along with its true expected reward $\theta_k$. Hollow dots indicate that we pulled the arm but received no reward. Solid dots indicate that a reward was given by the bandit. 

<video src="https://github.com/gdmarmerola/gdmarmerola.github.io/blob/master/assets/img/ts_for_mab/mab_1.mp4" height="320" controls preload></video>
