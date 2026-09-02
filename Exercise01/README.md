# Exercise 1 - README

Exercise 1 contains a function named:
```python
def local_regression_one_predictor(y, x, k, x0):
...
return pred, se
```
that takes in four variables: 
- $\vec{y}$, a vector of observations of the response variable,
- $\vec{x}$, a vector of observations of the predictor,
- $k$, the distance to furthest neighboring observations of $\vec{x}$ to include in each local gregression,
- $\vec{x_0}$, a vectir if values for which a predictor is going to be made,

and the function returns

- $\vec{pred}$, a vector of predicted values,
- $\vec{se}$, a vector of standard deviations of the expected value of each predicted value.

## Technical details
To make the local regression we use a linear model 
$f(x)=\hat{\beta_0} + \hat{\beta_1} x_1$
where we calculate $\\hat{\beta_0}$ and $\\hat{\beta_1}$ by solving for 
$\hat{\beta_0}, \hat{\beta_1} = \text{argmin}\hat{Q}$.
