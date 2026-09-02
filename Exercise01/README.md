# Exercise 1 - README

Exercise 1 contains a function named:
```python
def local_regression_one_predictor(y, x, k, x0):
...
return pred, se
```
that takes in four variables: 
- $\vec{y}$, a vector of observations of the response variable.
- $\vec{x}$, a vector of observations of the predictor.
- $k$, the distance to furthest neighboring observations of $\vec{x}$ to include in each local gregression.
- $\vec{x_0}$, a vectir if values for which a predictor is going to be made.

The function returns

- $\vec{pred}$, a vector of predicted values.
- $\vec{se}$, a vector of standard deviations of the expected value of each predicted value.
