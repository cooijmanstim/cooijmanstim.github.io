---
layout: default
title: New inpainting samples
date:   2017-04-09
---

### New inpainting samples

I've done a hyperparameter search and found that shallower models are drastically easier to optimize.
The thing with orderless NADE is that the random masking of the input adds a *lot* of noise to the optimization process, and as a result it is practically impossible to overfit anything.

I'm also exploring the use of discrete [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance#Computing_the_EMD) as the loss instead of the hopelessly categorical cross-entropy.
Optimization is *much* smoother with EMD, and I can imagine the gradient points toward more reasonable regions of hyperparameter space.
I'm running a new hyperparameter search with three different losses -- cross entropy and two variants of EMD with L1 and L2 distance -- in order to qualitatively compare the models they produce.

I've messed with residual and dilated convolutions but no particular luck. Dilated convolutions seem like a good fit though in order to bridge the 32x32 gap with fewer layers. The whole thing runs about 3x as fast with dilated convolutions, all else equal.

Below are some samples from a shallow EMD L1 model. First, Gibbs samples:

<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/13.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/5.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/4.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/11.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/12.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/7.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/6.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/17.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/15.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/19.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/1.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/3.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/8.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/2.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/10.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/0.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/9.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/16.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/18.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_independent_gibbs_2017-04-09T18:55:11.881425_T1.0/14.gif">

I was also interested to see how ancestral samples look. For ancestral sampling, one has to choose an ordering according to which to sample.
I tried three cases: orderless (random) ordering, greedy ordering (sample the variable with the lowest entropy first), antigreedy ordering (sample the variable with the highest entropy first).

Antigreedy ordering seems like a strange thing to do, but in our music work we found that this ordering got dramatically better log likelihood scores when *evaluating* validation data points.
I had a hunch this is an artifact of teacher-forcing, where after making its prediction the model would get to see the ground truth for that variable; *of course* you want to get the ground truth of the things you're most uncertain about.
However I was never sure whether it would also help during generation.

Orderless:

<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/13.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/5.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/4.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/11.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/12.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/7.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/6.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/17.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/15.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/19.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/1.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/3.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/8.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/2.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/10.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/0.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/9.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/16.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/18.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_orderless_ancestral_2017-04-09T05:25:22.139241_T1.0/14.gif">

Greedy:

<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/13.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/5.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/4.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/11.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/12.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/7.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/6.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/17.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/15.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/19.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/1.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/3.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/8.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/2.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/10.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/0.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/9.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/16.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/18.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_greedy_ancestral_2017-04-09T01:13:12.489489_T1.0/14.gif">

Antigreedy:

<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/13.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/5.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/4.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/11.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/12.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/7.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/6.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/17.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/15.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/19.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/1.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/3.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/8.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/2.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/10.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/0.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/9.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/16.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/18.gif">
<img src="/assets/images/sample_2017-04-09/sample_firstemd_antigreedy_ancestral_2017-04-09T03:19:25.120414_T1.0/14.gif">

