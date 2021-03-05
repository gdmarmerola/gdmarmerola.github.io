---
layout: post
title: The features and shortcomings of Bayesian Networks (with CausalNex)
featured-img: features_shortcomings_causalnex_cover
image: features_shortcomings_causalnex_cover
category: [causal inference]
mathjax: true
summary: Benchmarking Bayesian Networks (CausalNex) on a synthethic causal inference dataset
---

Bayesian Networks and structural causal models are really exciting. These methods invite us to *think hard* about our business problem and our assumptions. Causal graphs make it easy for everyone (not just data scientists) reason about and criticize the model. This level of explainability and rigor are in a stark contrast with the black box fit / predict logic that is very much present in the data science industry.

However, the widespread adoption of these models in industry still has yet to happen, at least from my personal standpoint. Given the growing interest of the community and new tools backed by big industry names (such as [CausalNex](https://causalnex.readthedocs.io/en/latest/) by McKinsey / QuantumBlack), I thought it would be cool to try them out and check out features and shortcomings from an industry practitioner standpoint.

So, in this post, I'll report my first try working with Bayesian Networks, more specifically using the CausalNex library. We'll try to solve a [synthetic causal inference problem](https://fklearn.readthedocs.io/en/latest/examples/causal_inference.html) from Nubank's fklearn library (the same that I've already used in several other [posts](https://gdmarmerola.github.io/diagnosting-validity-causal-effects-trees/)). Having a synthetic dataset gives us the benefit of knowing the true treatment effect and counterfactuals, providing objective evidence on the performance of the algorithm, even if it does not carry the weight of a thorough benchmark. 

Throughout the post, I'll try to point out the **features** and **shortcomings** of our Bayesian Networks / CausalNex approach. Without further ado, let's dive in!


## Data: `make_confounded_data` from `fklearn`

Nubank's `fklearn` module provides a nice synthetic causal inference problem generator, so we're going to use the same data generating process and example from its [documentation](https://fklearn.readthedocs.io/en/latest/examples/causal_inference.html). 

```python
# generating 
df_rnd, df_obs, df_cf = make_confounded_data(50000)
df_obs.head(5)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_0.png" | absolute_url }})

We have five features: `sex`, `age`, `severity`, `medication` and `recovery`. We want to estimate the impact of `medication` on `recovery`. So, our *target* variable is `recovery`, our *treatment* variable is `medication` and the rest are our *explanatory* variables.

A good counterfactual model will tell us how would be the recovery time for each individual for both potential decisions of taking or not taking medication. The model should be robust to confounders, variables that affect the probability of someone taking the medication. For instance, people with higher `severity` may be more likely to take the medicine. If not properly taken into account, this confounder may lead us to conclude that the medication may make recovery worse: people that took the medication may have recovered more slowly (effectively ignoring that their condition was also more severe in the first place). In the [fklearn's documentation](https://fklearn.readthedocs.io/en/latest/examples/causal_inference.html), the data generating process is shown in detail, highlighting the confounders in the data. The effect we're looking for is 

$$ \frac{E[recovery \vert medication = 1]}{E[recovery \vert medication = 0]} = exp(-1) = 0.368$$

The `make_confounded_data` function outputs three data frames: `df_rnd`, where treatment assingment is random, `df_obs`, where treatment assingment is confounded and `df_cf`, which is the counterfactual dataframe, containing the counterfactual outcome for all the individuals.

## 1. Structure

We can divide our causal inference problem in two steps:

1. Define (or learn) the *structure* of the problem. The structure determines which variables are causes of others and is represented by a causal diagram: a directed acyclic graph representing these relationships.
2. Given a structure and data, compute the counterfactual (treatment effect) of taking the medication vs. not taking it.

Let us start with step (1).

### True (Oracle) Structure

As we're dealing with a simulated dataset, we have the luxury of knowing the true structure of the problem. It is very easy to define this through the CausalNex `StructureModel` API:

```python
from causalnex.structure import StructureModel
sm_true = StructureModel()

sm_true.add_edges_from([
    ('age', 'severity'),
    ('sex', 'medication'),
    ('severity', 'medication'),
    ('medication', 'recovery'),
    ('sex', 'recovery'),
    ('age', 'recovery'),
    ('severity', 'recovery'),
])

plot_structure(sm_true)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_1.png" | absolute_url }})

If you're still confused about structural causal models now things may get clearer. The arrow sets the direction of causality, such that `a -> b` means that `a` *causes* `b`. So in our case `recovery` (our target, in red) is caused by all variables. The treatment variable, `medication`, is caused by `sex` and `severity`, and `severity` is caused by `age`. 

Remeber that this is a simulated dataset. We know the true diagram because we created it and defined the data generating process (in detail [here](https://fklearn.readthedocs.io/en/latest/examples/causal_inference.html)). In an industry application, we would never be sure of the diagram and would count on domain experts to try to make it reproduce reality as much as we can.

In any case, having the possibility of reasoning about the structure of the problem is clearly a **feature** to the practitioner!

### Trying to learn structure from data

CausalNex provides the ability to infer the structure from data, through the NOTEARS algorithm. The algorithm receives our data and a threshold to remove weak edges. 

```python
from causalnex.structure.notears import from_pandas

sm_learned = from_pandas(df_rnd)
sm_learned.remove_edges_below_threshold(0.5)

plot_structure(sm_learned)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_2.png" | absolute_url }})

Not bad, as the algorithm found that recovery depends on all variables. However, other edges seem awkward, specially `medication` being a cause of `age`. Nevertheless, structure learning is recognized to be a very hard problem, and in most cases a domain expert will step in and make corrections to the relationships. It is a potentially very useful **feature** for exploring the data.

For the sake of fun and benchmarking, we'll store this structure so we can test it down the road.

### Naive structure

Another way to try to define structure is to use the design pattern of `X` (features), `W` (treatment) and `y` (target). An example is the [Genearlized Random Forest](https://github.com/grf-labs/grf) method by Athey et. al. We assume the causal graph where `X -> W -> y` and `X -> y`. Each feature is assumed to be a potential cause of the target variable and the treatment variable as well.

```python
from causalnex.structure import StructureModel
sm_naive = StructureModel()

sm_naive.add_edges_from([
    ('medication', 'recovery'),
    ('sex', 'recovery'),
    ('age', 'recovery'),
    ('severity', 'recovery'),
    ('sex', 'medication'),
    ('age', 'medication'),
    ('severity', 'medication'),
])

plot_structure(sm_naive)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_3.png" | absolute_url }})

Cool. We're off to a good start. We have three structures to test and perform inference on. Moving on to step (2)!

## 2. Inference

We divide Inference in three substeps:

* **Dataprep:** we need to discretize our data so the Bayesian Network can perform computations. How we perform discretization makes a lot of difference, as we'll see further

* **Fit and evaluation:** we fit the network to data (which amounts to computing tables of probabilities) and evaluate it on a test set

* **Counterfactuals:** we use the network to perform do-calculus and use it to compute counterfactuals

We will walk through a single run with the true causal structure. Then, we'll devise a procedure to test other structures and hyperparameters. Finally, we'll analyze results.

### Dataprep

Bayesian Networks in general require that we discretize variables prior to fitting them to data. This can either be a shortcoming or a feature: it may make results more interpretable by business stakeholders. However, it introduces an arbritrary choice that may be hard to reason about and has a high impact on results. Personally, I think the scale is tipped to this being a **shortcoming**, as I think the interpretability gains do not compensate the introduction of an arbitrary choice.

For simplicity, we perform quantile-based discretization, where all the features are discretized to `n_bins` bins. We also transform binary variables into strings, log1p-transform the target variable, and split data into `train` and `test` sets.

We perform special treatment on the target variable such as the the discretized labels represent the expected value of the bins, so we preserve expected values and can correctly compute treatment effects further on. 

We start on the "easy mode" with `df_rnd`, where we do not have confounding.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def discretize(df, n_bins=8):
    
    target = df['recovery']
    continuous = df[['age', 'severity']]
    binary = df[['medication', 'sex']]
    
    target = (
        target
        .to_frame()
        .assign(recovery_bin = lambda x: pd.qcut(x['recovery'], n_bins, duplicates='drop'))
        .groupby('recovery_bin').transform(lambda x: x.mean())
        .apply(np.log1p)
    
    )
    
    continuous = (
        continuous
        .apply(lambda x: pd.qcut(x, n_bins, duplicates='drop'))
        .apply(lambda x: [e.mid for e in x])
    )
    
    binary = binary.astype(int)
    binary['sex'] = binary['sex'].replace({1:'female', 0:'male'})
    binary['medication'] = binary['medication'].replace({1:'yes', 0:'no'})
    
    return pd.concat([continuous, binary, target], axis=1)

df_disc = discretize(df_rnd)
train, test = train_test_split(df_disc, train_size=0.8, random_state=42)
```

### Fitting and evaluating the Bayesian Network

We then proceed to fit a Bayesian Network to our processed data. `.fit_node_states(...)` acts like a dictionary / schema, saving the possible states of the network. Then, `.fit_cpds(...)` computes conditional probabilities. Now we transformed our structure into a generative model that can answer probabilistic queries.

```python
from causalnex.network import BayesianNetwork

bn_true = BayesianNetwork(sm_true)

bn_true = bn_true.fit_node_states(df_disc)
bn_true = bn_true.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
```

The first thing we do is to run our test  set thorugh the net and see if it can recover the true states of the target variable. This is actually a multi-class classification problem as we discretized our variables, so we use the micro-average ROC AUC.

```python
from causalnex.evaluation import roc_auc
roc, auc = roc_auc(bn_true, test, "recovery")
print(auc)
```

```
0.9284841721428572
```

This is actually a high AUC! The network did a good job computing de conditional probability distributions of the data. For instance, we can check that the probability of taking medication remains constant after conditioning on `severity` and `sex`, which is expected as we're using `df_rnd`. That kind of interpretability is clearly a **feature**.

```python
bn_true.cpds['medication']
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_4.png" | absolute_url }})

Let us now move to computing counterfactuals!

### Computing counterfactuals

As the network can act as a generative model, we can compute probabilistic queries on it. Furthermore, being a structural causal model, the net can also perform do-calulus $P(y \vert do(X))$ and compute counterfactuals. Specifically, we're interested in

$$T = \frac{P(recovery \vert do(medication = 1))}{P(recovery \vert do(medication = 0))}$$

where the expected value of $T$ is $exp(-1) = 0.368$. 

We start by using the `InferenceEngine` class, which exposes the `.do_intervention(...)` method, allowing us to perform do calculus. The following function returns the distribution of `recovery` given no treatment (`0`) and treatment (`1`). It is fairly easy and elegant to use and a good **feature** to have.

```python
from causalnex.inference import InferenceEngine

def compute_medication_cf(bn):

    ie = InferenceEngine(bn)

    ie.do_intervention("medication", {'yes': 0.0, 'no': 1.0})
    recovery_no = pd.Series(ie.query()["recovery"])

    ie.do_intervention("medication", {'yes': 1.0, 'no': 0.0})
    recovery_yes = pd.Series(ie.query()["recovery"])

    cf = pd.concat([recovery_no, recovery_yes], axis=1)
    cf.index = np.expm1(cf.index)
    
    return cf

counterfactuals = compute_medication_cf(bn_true)
counterfactuals
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_5.png" | absolute_url }})

We can plot the distribution to check how the `medication` impacts `recovery`:

```python
def plot_counterfactuals(cf):

    plt.figure(figsize=(12, 5), dpi=120)

    plt.plot(cf.index, cf[0], 'o--', label='medication = no')
    plt.fill_between(cf.index, cf[0], alpha=0.1)

    plt.plot(cf.index, cf[1], 'o--', label='medication = yes')
    plt.fill_between(cf.index, cf[1], alpha=0.1)

    plt.legend()
    plt.title('Distribution of recovery given medication')
    plt.xlabel('recovery');
    plt.ylabel('Probability');
    
plot_counterfactuals(counterfactuals)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_6.png" | absolute_url }})

We can clearly see that the distribution of recovery times for `medication = yes` is fairly skewed to the left as compared to `medication = no`, which shows an opposite pattern, indicating that medication provides an improvement. 

Then, we compute the effects by calculating the expected `recovery` value given treatment and no treatment. This is easy as we're assured that `recovery` labels reflect expected values at each bin.

```python
def compute_effect(cf):

    expected_no = (cf.index * cf[0]).sum()
    expected_yes = (cf.index * cf[1]).sum()
    
    effect = expected_yes/expected_no
    
    return effect

effect = compute_effect(counterfactuals)
print(f'The effect computed is {effect:.4f} / {np.log(effect):.2f} (log)')
```

```
The effect computed is 0.3975 / -0.92 (log)
```

And we're off to a reasonable start! The net recovered an effect of `0.3975` or `-0.92` in log space, which is close to the expected `0.368` and `-1` true values. The ease of use and the fact that the net recovered the true effect counts as a **feature**. 

However, there's an important **shortcoming**: the net does not provide confidence intervals. A Boostrap procedure could solve this, but it comes with the additional caveat of repeating the computationally expensive processes of training and executing do-calculus for several nets.

### Impact of confounding

Let us now try to solve `df_obs`, which has a high degree of confounding. For simplicity, let us abstract all steps into the `run_full_bayes_net_test` function:

```python
def run_full_bayes_net_test(df, structure, n_bins, verbose=1):
        
    ## preparing data ##
    
    df_disc = discretize(df, n_bins)
    train, test = train_test_split(df_disc, train_size=0.8, random_state=42)
    
    ## setting up network and inference engine ##
    
    bn = BayesianNetwork(structure)
    
    ## fitting and evaluating network ##
    
    bn = bn.fit_node_states(df_disc)
    bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
    
    roc, auc = roc_auc(bn, test, "recovery")
    
    ## running counterfactuals via inference engine ##
    
    counterfactuals = compute_medication_cf(bn)
    effect = compute_effect(counterfactuals)
    
    if verbose > 0:
        plot_counterfactuals(counterfactuals)
        print(f'The effect computed is {effect:.4f} / {np.log(effect):.2f} (log)')    
    
    return {
        'auc': auc,
        'effect': effect,
    }
```

Now we can run our test with any dataframe, structure and number of bins. Let us check out what happens if we run the net just like before but using the confounded `df_obs` dataset:


```python
run_full_bayes_net_test(df_obs, sm_true, 8)
```

```
The effect computed is 0.7656 / -0.27 (log)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_7.png" | absolute_url }})

Actually, in this case, confounding damaged the counterfactual estimate of the network, which underestimated the effect by a large margin (`0.7656 ` vs. `0.368`/ `-0.27` vs. `-1.0`), despite still getting the direction right (medication improves recovery). What did go wrong? I actually expected that the Bayesian Network would perform well on the confounded case, and would count this as a major **shortcoming**. 

Let us move to test how different hyperparmeters impact the performance of the Bayesian Network. Particularly, we'll examime `n_bins` and `structure`, for both confounded and unconfouded datasets.

### Impact of hyperparameters and confounding

To test different hyperparamters the code below performs a grid search where we try `n_bins` in `[2, 4, 8, 16, 32]`, `structure` in `[sm_true, sm_learned, sm_naive]` and `df` in `[df_obs, df_rnd]`. 

```python
import itertools

result_list = []

grid_bins = [2, 4, 8, 16, 32]
grid_df = [('df_obs', df_obs), ('df_rnd', df_rnd)]
grid_structure = [('true', sm_true), ('learned', sm_learned), ('naive', sm_naive)]

for n_bins, df, structure in itertools.product(grid_bins, grid_df, grid_structure):
    
    result = run_full_bayes_net_test(df[1], structure[1], n_bins, verbose=0)
    
    result['n_bins'] = n_bins
    result['dataset'] = df[0]
    result['structure'] = structure[0]
    
    result_list.append(result)

plot_results(result_list)
```

![]({{ "assets/img/features_shortcomings_causalnex/causalnex_8.png" | absolute_url }})

The plots show a interesting pattern. The true effect is shown by the black dashed line at `y = -1.0`. In the uncounfounded case, our counterfactual estimation procedure got really close to the true effect using an intermediate number of bins (4 and 8). The best structure was surprisingly the naive structure, although not by a large margin. The learned structure and the true structure arrived at almost identical results, which is also puzzling. 

However, we see another major **shortcoming** of Bayesian Networks (at least in this setting): if the number of bins is too high or too low, we greatly underestimate the treatment effect. In a real-world setting, where we would not know the true effect, tuning this parameter would be challenging. Moreover, I would find very hard to trust the treatment effect estimate knowing of this sensibility to hyperparameters.

In the confounded case we observe the same pattern as we did before, with our Bayesian Network approach underestimating the effect across the board, independently of structure or discretization choice.

AUC results are the same independent of structure, which is puzzling as well (maybe there's a bug on my code?).

Let us now summarize all that we've learned and list the main **features** and **shortcomings** of Bayesian Networks.

## Conclusion

In this post, we took CausalNex for a spin, with the goal to test a promising method for causal inference, Bayesian Networks. We used a synthetic causal inference dataset to allow us to compare the model results to ground truth in an objective way. From this exploration we could check the main **features** and **shortcomings** of the approach. I list them below.

#### *Features* 

* **Building the DAG:** having a interface for a structural causal model incentivizes the practitioner to *think hard* about the problem and invites contribution from everyone (not just data scientists). We can also blend data with domain knowledge.

* **Structure learning:** even if we did not recover the true structure exactly, the structure learning functionaly did a decent job and could be a valuable tool for exploring a new dataset.

* **Inspecting conditional probabilities:** our Bayesian Network went one step further in interpretability when showing its table of conditional probabilities. I would argue that `.cpds` provides an overview of the relationships between variables in a much more powerful way than correlation analysis.

* **Do-calculus:** the do-calculus functionality is very easy to use and flexible. 

#### *Shortcomings* 

* **Discretization:** having the need of discretizing the data can be a feature to make the dataset easier to reason about. However, as we've seen, the introduction of this arbitrary choice can have a great impact on the counterfactual estimates.

* **No confidence intervals, and high computational cost:** we lacked confidence intervals, which are very useful to diagnose the accuracy of the estimated effects. One could argue that this can be solved with bootstrap, but the high computational cost of the Bayesian Network can make this solution unfeasible.

* **High hyperparameter sensivity:** small changes on hyperparameters greatly impacted the treatment effect estimate from the net (specially the ones related to discretization).

* **Poor performance on confounded data:** the implementation showed poor results on confounded data. 

And that's it! I hope it was a good read. All feedbacks are highly appreciated!
