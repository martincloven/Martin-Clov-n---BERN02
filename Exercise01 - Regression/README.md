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
- &nbsp; $k$, the distance to furthest neighboring observations of $\vec{x}$ to include in each local gregression,
- $\vec{x_0}$, a vectir if values for which a predictor is going to be made,

and the function returns

- $\vec{pred}$, a vector of predicted values,
- $\vec{se}$, a vector of standard deviations of the expected value of each predicted value.

## Technical details
For our local regression we assume that our data follows a linear equation of $f(x)=\beta_0 + \beta_1 x$.

We use the "hat" sign as in e.g., $\hat{\beta_0}$ to denote our estimates of parameters allowing us to see if we are refering to the true values of a parameter or our derived estimations of them.

By finding the smallest residual sum of squares (RSS) over a large variety of possible combinations of $\hat{\beta_0}$ and $\hat{\beta_1}$ values we will get the estimated parameters $\hat{\beta_0}$ and $\hat{\beta_1}$ that closest describe our observed data. Therefore we get 

$\hat{\beta_0}, \hat{\beta_1} = \underset{\beta_0 , \beta_1}{\text{argmin}}(Q)$ 

using

$Q = \displaystyle\sum_{i=1}^{\beta_{max}} \displaystyle\sum_{j=1}^{\beta_{max}} \displaystyle\sum_{g=1}^{n}  = \omega_g(y_k - \beta_{0i} - \beta_{1j}x_g)^2 $

where $\omega_g$ is an appropriate weight assigned to $(x_g,y_g)$, $n$ is the nr. of points in the range $x_0 \pm k$ and as of release, the number of values for both $\hat{\beta_0}$ and $\hat{\beta_1}$ are $\beta_{max} = 400$ with values going from -1000 to 1000.

To get our weights $\omega_g$ we use 

$f(x) = e^{-Cx^2}$ where $C = -\ln(0.005)/k^2$ 

which means that the height $f(x_0 \pm k) = 0.005$,
ensuring that all neighboring points with at most $k$ units away in the x-axis gets a large weight if close to $x_0$ which steadily decreases as we go towards $x_0 \pm k$. We also put all values that are further than $k$ units away from $x_0$ to zero. Having $k$ as an variable of distance instead of number of points away from $x_0$ helps with giving ease to the user if there are upwards of thousands observation points, making it otherwise much difficult to pick a precise range to choose from and the same holds if there are very few observations at a location, making it required to observe the data by eye each time you want to use the function at a new $x_0$ location.

With appropriate values for $\hat{\beta_0}$ and $\hat{\beta_1}$ we calculate our prediction $pred$ at $x_0$ by

$pred = \hat{\beta_0} + \hat{\beta_1}x_0$.

To get the standard deviation $se$ we use our calculated $\hat{\beta_0}$ and $\hat{\beta_1}$ values at each non zero weighted $x$ observation and calculate a new array of $\hat{y}$ values that we then use to get the difference $e_i$ from the observed response variables $y$, $e_i = y_i - \hat{y_i} = y_i - \hat{\beta_0} - \hat{\beta_1}x_{i1}$ resulting in a standard deviation $se$ of 

$se = \sqrt{\frac{1}{n-2}\displaystyle\sum_{i=1}^{n} e_i^2}$.

The two degrees of freedom utilized as $\hat{\beta_0}$ and $\hat{\beta_1}$ make $p=2$ for $\frac{1}{n-p}$ giving us the $\frac{1}{n-2}$.


The code is built so that it can take in multiple different $x_0$ positions resulting in a prediction $pred$ with a standard deviation $se$ for each one given in order. For each new $x_0$, a new pair of $\hat{\beta_0}$ and $\hat{\beta_1}$ are calculated as well.
 
## Possible improvements
A second loop can be built to get more accurate values for $\hat{\beta_0}$ and $\hat{\beta_1}$ by looping over an additional $\beta_{max} = 400$ values going from the closest neighboring two values to the one picked.

Give a weighted value $\omega_e$ to each prediction point difference $e_i$ calculated for the standard deviation $se$
