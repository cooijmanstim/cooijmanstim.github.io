---
layout: blog
title: "Tweaking the generative process"
date:   2017-04-23
category: blog
---

### Tweaking the generative process

These are samples from the same model as those in [this post](https://cooijmanstim.github.io/2017/04/21/more-contiguous-masking-results/). The only thing that's different is that I've messed with the distribution temperature and the gibbs annealing schedule (the schedule from [this paper](https://arxiv.org/abs/1409.0585) but with different hyperparameters).

Changing the temperature of a categorical distribution reduces or increases its entropy; the temperature can be used to interpolate between a deterministic ($T = 0$) or uniform ($T = \infty$) distribution. Temperature is implemented by raising the probability to a power $T$ and then renormalizing:

$$\widetilde{p} \propto p^T$$

The rationale for increasing the temperature is because models trained by maximum likelihood tend to be uncertain and spread probability around where they shouldn't. Certainly if we care about getting high-quality samples rather than likelihood, we'd like to sample from a peakier distribution.

Upping the temperature of conditional distributions in autoregressive models isn't quite justified though. You want to increase the temperature of the joint, not of distributions over individual variables. In the limit $T \to 0$, the distribution becomes a detemrinistic argmax, and if you do this one variable at a time you're basically sampling each variable greedily. Instead, people use things like beam search if they want to find high-probability samples.

Here's how changing the temperature affects the samples:

<table>
<thead>
<th>Temperature</th>
<th>Samples</th>
</thead>
<tbody>

<tr><td>0.99</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.99_independent_gibbs_2017-04-22T16:10:26.661541_T0.99/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.99_independent_gibbs_2017-04-22T16:10:26.661541_T0.99/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.99_independent_gibbs_2017-04-22T16:10:26.661541_T0.99/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.99_independent_gibbs_2017-04-22T16:10:26.661541_T0.99/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.99_independent_gibbs_2017-04-22T16:10:26.661541_T0.99/3.gif">
</td></tr>

<tr><td>0.90</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.9_independent_gibbs_2017-04-22T21:59:21.354625_T0.9/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.9_independent_gibbs_2017-04-22T21:59:21.354625_T0.9/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.9_independent_gibbs_2017-04-22T21:59:21.354625_T0.9/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.9_independent_gibbs_2017-04-22T21:59:21.354625_T0.9/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.9_independent_gibbs_2017-04-22T21:59:21.354625_T0.9/3.gif">
</td></tr>

<tr><td>0.10</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.1_independent_gibbs_2017-04-23T03:47:09.335287_T0.1/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.1_independent_gibbs_2017-04-23T03:47:09.335287_T0.1/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.1_independent_gibbs_2017-04-23T03:47:09.335287_T0.1/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.1_independent_gibbs_2017-04-23T03:47:09.335287_T0.1/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0.1_independent_gibbs_2017-04-23T03:47:09.335287_T0.1/3.gif">
</td></tr>

<tr><td>0.00</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0_independent_gibbs_2017-04-26T05:47:24.008595_T0.0/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0_independent_gibbs_2017-04-26T05:47:24.008595_T0.0/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0_independent_gibbs_2017-04-26T05:47:24.008595_T0.0/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0_independent_gibbs_2017-04-26T05:47:24.008595_T0.0/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_pmin0.0001_T0_independent_gibbs_2017-04-26T05:47:24.008595_T0.0/3.gif">
</td></tr>

</tbody>
</table>

The Gibbs annealing schedule from [Yao et al](https://arxiv.org/abs/1409.0585) is a truncated linear schedule. If $n$ is the index of the current Gibbs step, and $N$ is the total number of Gibbs steps to be taken, then each variable is resampled with probability

$$\alpha_n = max(\alpha_{min}, \alpha_{max} - \frac{n}{N} \frac{\alpha_{max} - \alpha_{min}}{\eta})$$

Basically, if $\eta = 1$, it starts at $\alpha_{max}$ and drops down linearly to $\alpha_{min}$. If $\eta < 1$, it drops down faster and becomes constant as soon as it crosses $\alpha_{min}$.

In my previous music work the default schedule of $\alpha_{max} = 0.9$, $\alpha_{min} = 0.1$ and $\eta = 0.7$ seemed to just work. In the image inpainting case it seems like there are waaay too may variables to sample, and that much more time should be spent resampling smaller subsets of variables. Smaller subsets means more context means more information means less entropy, supposedly, so this should have a similar effect as reducing temperature. However, our Gibbs procedure has a backtracking flavor to it, as the model gets to revisit previous decisions.

Here's what it does ($\alpha_{max} = 0.5$ everywhere):

<table>
<thead>
<th>Temperature</th>
<th>$\alpha_{min}$</th>
<th>$\eta$</th>
<th>Samples</th>
</thead>
<tbody>

<tr><td>1.0</td><td>0.1</td><td>0.1</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T06:17:38.057296_T1.0_yao0.100000,0.500000,0.100000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T06:17:38.057296_T1.0_yao0.100000,0.500000,0.100000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T06:17:38.057296_T1.0_yao0.100000,0.500000,0.100000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T06:17:38.057296_T1.0_yao0.100000,0.500000,0.100000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T06:17:38.057296_T1.0_yao0.100000,0.500000,0.100000/3.gif">
</td></tr>

<tr><td>1.0</td><td>0.1</td><td>0.001</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T07:33:14.725483_T1.0_yao0.100000,0.500000,0.001000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T07:33:14.725483_T1.0_yao0.100000,0.500000,0.001000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T07:33:14.725483_T1.0_yao0.100000,0.500000,0.001000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T07:33:14.725483_T1.0_yao0.100000,0.500000,0.001000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T07:33:14.725483_T1.0_yao0.100000,0.500000,0.001000/3.gif">
</td></tr>

<tr><td>1.0</td><td>0.001</td><td>0.1</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T08:48:21.753076_T1.0_yao0.001000,0.500000,0.100000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T08:48:21.753076_T1.0_yao0.001000,0.500000,0.100000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T08:48:21.753076_T1.0_yao0.001000,0.500000,0.100000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T08:48:21.753076_T1.0_yao0.001000,0.500000,0.100000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T08:48:21.753076_T1.0_yao0.001000,0.500000,0.100000/3.gif">
</td></tr>

<tr><td>1.0</td><td>0.001</td><td>0.001</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T10:03:36.678420_T1.0_yao0.001000,0.500000,0.001000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T10:03:36.678420_T1.0_yao0.001000,0.500000,0.001000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T10:03:36.678420_T1.0_yao0.001000,0.500000,0.001000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T10:03:36.678420_T1.0_yao0.001000,0.500000,0.001000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T10:03:36.678420_T1.0_yao0.001000,0.500000,0.001000/3.gif">
</td></tr>

<tr><td>1.0</td><td>0.00001</td><td>0.1</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T11:19:26.772963_T1.0_yao0.000010,0.500000,0.100000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T11:19:26.772963_T1.0_yao0.000010,0.500000,0.100000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T11:19:26.772963_T1.0_yao0.000010,0.500000,0.100000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T11:19:26.772963_T1.0_yao0.000010,0.500000,0.100000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T11:19:26.772963_T1.0_yao0.000010,0.500000,0.100000/3.gif">
</td></tr>

<tr><td>1.0</td><td>0.00001</td><td>0.001</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T12:34:30.923041_T1.0_yao0.000010,0.500000,0.001000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T12:34:30.923041_T1.0_yao0.000010,0.500000,0.001000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T12:34:30.923041_T1.0_yao0.000010,0.500000,0.001000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T12:34:30.923041_T1.0_yao0.000010,0.500000,0.001000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T12:34:30.923041_T1.0_yao0.000010,0.500000,0.001000/3.gif">
</td></tr>



<tr><td>0.1</td><td>0.1</td><td>0.1</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T13:50:07.698572_T0.1_yao0.100000,0.500000,0.100000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T13:50:07.698572_T0.1_yao0.100000,0.500000,0.100000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T13:50:07.698572_T0.1_yao0.100000,0.500000,0.100000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T13:50:07.698572_T0.1_yao0.100000,0.500000,0.100000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T13:50:07.698572_T0.1_yao0.100000,0.500000,0.100000/3.gif">
</td></tr>

<tr><td>0.1</td><td>0.1</td><td>0.001</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T19:40:31.060057_T0.1_yao0.100000,0.500000,0.001000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T19:40:31.060057_T0.1_yao0.100000,0.500000,0.001000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T19:40:31.060057_T0.1_yao0.100000,0.500000,0.001000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T19:40:31.060057_T0.1_yao0.100000,0.500000,0.001000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-27T19:40:31.060057_T0.1_yao0.100000,0.500000,0.001000/3.gif">
</td></tr>

<tr><td>0.1</td><td>0.001</td><td>0.1</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T12:08:57.900827_T0.1_yao0.001000,0.500000,0.100000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T12:08:57.900827_T0.1_yao0.001000,0.500000,0.100000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T12:08:57.900827_T0.1_yao0.001000,0.500000,0.100000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T12:08:57.900827_T0.1_yao0.001000,0.500000,0.100000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T12:08:57.900827_T0.1_yao0.001000,0.500000,0.100000/3.gif">
</td></tr>

<tr><td>0.1</td><td>0.001</td><td>0.001</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T15:05:48.312154_T0.1_yao0.001000,0.500000,0.001000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T15:05:48.312154_T0.1_yao0.001000,0.500000,0.001000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T15:05:48.312154_T0.1_yao0.001000,0.500000,0.001000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T15:05:48.312154_T0.1_yao0.001000,0.500000,0.001000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T15:05:48.312154_T0.1_yao0.001000,0.500000,0.001000/3.gif">
</td></tr>

<tr><td>0.1</td><td>0.00001</td><td>0.1</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T18:01:29.001650_T0.1_yao0.000010,0.500000,0.100000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T18:01:29.001650_T0.1_yao0.000010,0.500000,0.100000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T18:01:29.001650_T0.1_yao0.000010,0.500000,0.100000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T18:01:29.001650_T0.1_yao0.000010,0.500000,0.100000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T18:01:29.001650_T0.1_yao0.000010,0.500000,0.100000/3.gif">
</td></tr>

<tr><td>0.1</td><td>0.00001</td><td>0.001</td><td>
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T20:58:28.633670_T0.1_yao0.000010,0.500000,0.001000/4.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T20:58:28.633670_T0.1_yao0.000010,0.500000,0.001000/0.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T20:58:28.633670_T0.1_yao0.000010,0.500000,0.001000/2.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T20:58:28.633670_T0.1_yao0.000010,0.500000,0.001000/1.gif">
<img src="/assets/images/sample_2017-04-23/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-28T20:58:28.633670_T0.1_yao0.000010,0.500000,0.001000/3.gif">
</td></tr>

It's getting better, but I feel like I'm reaching the edge of what this model has learned to do.

</tbody>
</table>
