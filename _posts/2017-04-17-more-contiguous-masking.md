---
layout: default
title: More contiguous masking
date:   2017-04-17
---

### More contiguous masking

As promised in the last post, I've been working on more contiguous forms of masking to de-emphasize trivial local correlations when training the model. I've tried masking a randomly positioned 32x32 rectangle, just like I do at generation time, but the results are terrible and I feel like it's too much. Here are some samples from a model trained on squared EMD with 32x32 rectangular masks:

<table>
<thead>
<th>Strategy</th>
<th>Samples</th>
</thead>
<tbody>

<tr><td>Greedy</td><td>
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_greedy_ancestral_2017-04-12T19:07:03.690342_T1.0/13.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_greedy_ancestral_2017-04-12T19:07:03.690342_T1.0/5.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_greedy_ancestral_2017-04-12T19:07:03.690342_T1.0/4.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_greedy_ancestral_2017-04-12T19:07:03.690342_T1.0/11.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_greedy_ancestral_2017-04-12T19:07:03.690342_T1.0/12.gif">
</td></tr>

<tr><td>Antigreedy</td><td>
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_antigreedy_ancestral_2017-04-12T21:30:12.346989_T1.0/7.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_antigreedy_ancestral_2017-04-12T21:30:12.346989_T1.0/6.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_antigreedy_ancestral_2017-04-12T21:30:12.346989_T1.0/17.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_antigreedy_ancestral_2017-04-12T21:30:12.346989_T1.0/15.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_antigreedy_ancestral_2017-04-12T21:30:12.346989_T1.0/19.gif">
</td></tr>

<tr><td>Orderless</td><td>
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_orderless_ancestral_2017-04-12T17:31:26.129887_T1.0/1.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_orderless_ancestral_2017-04-12T17:31:26.129887_T1.0/3.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_orderless_ancestral_2017-04-12T17:31:26.129887_T1.0/8.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_orderless_ancestral_2017-04-12T17:31:26.129887_T1.0/2.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_orderless_ancestral_2017-04-12T17:31:26.129887_T1.0/10.gif">
</td></tr>

<tr><td>Gibbs</td><td>
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_independent_gibbs_2017-04-12T15:56:50.039729_T1.0/0.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_independent_gibbs_2017-04-12T15:56:50.039729_T1.0/9.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_independent_gibbs_2017-04-12T15:56:50.039729_T1.0/16.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_independent_gibbs_2017-04-12T15:56:50.039729_T1.0/18.gif">
<img src="/assets/images/sample_2017-04-12/sample_emd2_deepish_contiguous_independent_gibbs_2017-04-12T15:56:50.039729_T1.0/14.gif">
</td></tr>

</tbody>
</table>

There's another approach I'd like to explore, in which I stick with the idea of using independent Bernoullis to decide what to mask out, but instead of taking them as-is I mask out a whole neighborhood around each masked-out variable:

<iframe src="/assets/contiguous_masking.html" width="100%" height="1000"></iframe>

I mask out the neighborhood by taking a conjunction of translations of the Bernoulli mask. This increases the number of variables masked out, so in order for $p$ to have the same meaning I must use a transformed value $q$ such that

$$
p = q^a
$$

where $a$ is the area of the neighborhood (including the center).

There's another technicality to take care of. The masks for plain orderless NADE are chosen so that the size of the mask is uniformly distributed. This is done to ensure that conditional distributions with very many or very few variables in the condition (these would be rare under a binomial distribution) are sampled just as frequently as the more common ones.

An obvious way to enforce uniform mask size with plain Bernoullis is to sample a mask size $k$ and then use `np.random.choice` to decide which $k$ variables to mask out (or, as I've been doing in Tensorflow, create a mask with $k$ ones in the front and then shuffle it).

It's not obvious how to do this in the presence of potentially overlapping neighborhoods. However, a miracle has our backs.

As already stated, given Bernoulli masking probability $p$, the mask size $k$ follows a binomial distribution:

$$
\Pr(k | p) = \binom{n}{k} p^k (1 - p)^{n - k}
$$

If only we could find some prior distribution over $p$ to make the expectation uniform...

$$
\int_0^1 \Pr(k | p) \Pr(p) dp = \frac{1}{n + 1}
$$

where $n + 1$ is the number of possible values $k$ can take on. The miracle occurs when we let $Pr(p)$ be the uniform distribution on the unit interval. Then

$$
\begin{align}
\int_0^1 \Pr(k | p) \Pr(p) dp &= \int_0^1 \binom{n}{k} p^k (1 - p)^{n - k} dp
                           \\ &= \binom{n}{k} \int_0^1 p^k (1 - p)^{n - k} dp
                           \\ &= \binom{n}{k} B(k + 1, n - k + 1)
                           \\ &= \binom{n}{k} \frac{\Gamma(k + 1) \Gamma(n - k + 1)}{\Gamma(n + 2)}
                           \\ &= \frac{n!}{k!(n - k)!} \frac{k! (n - k)!}{(n + 1)!}
                           \\ &= \frac{1}{n + 1}.
\end{align}
$$

where $B$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function).

Putting everything together, I sample a probability $p \sim U(0, 1)$, compute $q \gets p^{1 / a}$, sample a mask according to independent Bernoullis with probability $q$, and conjoin $a$ local translations of the mask to obtain a contiguish mask.

Results to follow.
