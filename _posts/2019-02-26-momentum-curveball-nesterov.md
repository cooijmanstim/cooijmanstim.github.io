---
layout: blogpost
title: "Momentum, Curveball and Nesterov Acceleration"
date:   2019-02-26
category: blog
---

This post explores some relationships between the recent Curveball optimization method ([arXiv](https://arxiv.org/abs/1805.08095) or more recently [OpenReview](https://openreview.net/forum?id=Sygx4305KQ)) and Nesterov momentum (as popularized by [Sutskever et al., 2013](http://proceedings.mlr.press/v28/sutskever13.html)).

Suppose we are iteratively optimizing a scalar-valued function $f(x)$ wrt $x$ by gradient descent. We wish to take a step $z$ that minimizes $f(x+z)$, whose Taylor series is given by

$$f(x+z) = f(x) + z^\top f'(x) + \frac{1}{2} z^\top f''(x) z + \cdots$$

where $f'(x)$ denotes the gradient of $f$ evaluated at $x$, and similarly $$f''(x)$$ is the Hessian.

Second-order methods locally model $f(x)$ by a second-order approximation $\hat f$:

$$\hat f(x+z) = f(x) + z^\top f'(x) + \frac{1}{2} z^\top f''(x) z,$$

which has a single stationary point at

$$ z^\star = (f''(x))^{-1} f'(x)$$

If $$f''(x)$$ is PSD, i.e. $f$ is convex around $x$,
then $z^\star$ is the minimum of $\hat f$,
and hence the optimal step under this model.
However, multiplying by the inverse Hessian is expensive.

The authors of Curveball propose instead to solve for $z^\star$ by gradient descent,
and to interleave this inner optimization with the outer optimization over $x$:

$$
\begin{align}
z_{t+1} &= \rho z_t - \beta \frac{\partial \hat f}{\partial z}(x_t + z_t) \\
x_{t+1} &= x_t + z_{t+1}
\end{align}
$$

where $\rho,\beta$ are hyperparameters.
Note that $$\frac{\partial \hat f}{\partial z}(x+z) = f'(x) + f''(x) z$$; if the approximation were first-order instead, the $$f''(x)z$$ term would disappear and this would be equivalent to classical momentum.

But if we're going to use gradients to optimize $z$, then why make the second-order approximation at all? Why not just minimize $f(x+z)$ instead of $\hat f (x+z)$:

$$
\begin{align}
z_{t+1} &= \rho z_t - \beta f'(x_t + z_t) \\
x_{t+1} &= x_t + z_{t+1}.
\end{align}
$$

This update rule is practically equivalent to that for Nesterov momentum given by Sutskever et al ([2013](http://proceedings.mlr.press/v28/sutskever13.html)),
up to a scaling by $\rho$ on $z$ in the evaluation point of the gradient $f'$.

What does that mean? It means that if classical momentum is first-order and Curveball is second-order, why then Nesterov momentum must be the infinite-order version of momentum.

It's tempting to conclude that Curveball is a worse version of Nesterov momentum, as it makes an unnecessary second-order approximation. However, the Curveball paper goes on to adopt all the usual second-order tricks (GGN, trust region, etc.), which makes the comparison much less clear.

