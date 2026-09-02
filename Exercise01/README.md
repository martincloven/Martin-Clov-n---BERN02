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
$f(x)=\hat{\beta_0} + \hat{\beta_1} x$
and we calculate $\\hat{\beta_0}$ and $\\hat{\beta_1}$ by solving for 
$\hat{\beta_0}, \hat{\beta_1} = \text{argmin}(\hat{Q})$.

To get a good $\hat{\beta_0}$ and $\hat{\beta_1}$ pair of values we need check over a large variety of possible combination to get the right ones for

$\hat{Q} = \displaystyle\sum_{i=1}^{\beta_{max}} \displaystyle\sum_{j=1}^{\beta_{max}} \displaystyle\sum_{g=1}^{n}  = \omega_g(y_k - \beta_{0i} - \beta_{1j}x_g)^2 $

where $\omega_g$ is an appropriate weight assigned to $(x_g,y_g)$, $n$ is the nr. of points in the range $x_0 \pm k$ and as of release, the number of values for both $\hat{\beta_0}$ and $\hat{\beta_1}$ are $\beta_{max} = 400$ with values going from -1000 to 1000.

To get our weights $\omega_g$ we use 
$f(x) = e^{-Cx^2}$ where $C = -\ln(0.005)/k^2$ 
which means that the height $f(x_0 \pm k) = 0.005$,
ensuring that all neighboring points with at most $k$ units away in $\hat{x}$ gets a large weight if close to $x_0$ which steadily decreases as we get to $x_0 \pm k$. We also put all values that further than $k$ units away from $x_0$ to zero. Having $k$ as an variable of distance instead of number of points away from $x_0$ helps with giving ease to the user if there are upwards of thousands observation points, making it otherwise much difficult to pick a precise range to choose from and the same holds if there are very few observations at a location, making it required to observe the data by eye each time you want to use the function at a new $x_0$ location.

With appropriate values for $\hat{\beta_0}$ and $\hat{\beta_1}$ we calculate our prediction $pred$ at $x_0$ by
$pred = \hat{\beta_0} + \hat{\beta_1}x_0$.

To get the standard deviation $se$ we use our calculated $\hat{\beta_0}$ and $\hat{\beta_1}$ values at each non zero weighted $x$ observation and calculate a new array of $\hat{y}$ values that we then use to get the difference $e_i$ from the observed response variables $y$,
$e_i = y_i - $\hat{y_i}$ 
 
## Possible improvements
A second loop can be built to get more accurate values for $\hat{\beta_0}$ and $\hat{\beta_1}$ by looping over an additional $\beta_{max} = 400$ values going from the closest neighboring two values to the one picked.

Give a weighted value $\omega_e$ to each prediction point difference $e_i$ calculated for the standard deviation $se$
