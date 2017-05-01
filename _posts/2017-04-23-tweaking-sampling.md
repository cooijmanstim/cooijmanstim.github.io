---
layout: default
title: "Tweaking the generative process"
date:   2017-04-23
---

### Tweaking the generative process

These are samples from the same model as those in [this post](https://cooijmanstim.github.io/2017/04/21/more-contiguous-masking-results/). The only thing that's different is that I've messed with the distribution temperature and the gibbs annealing schedule (the schedule from [this paper](https://arxiv.org/abs/1409.0585) but with different hyperparameters).

Changing the temperature:

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

Changing the schedule ($p_max = 0.5$ everywhere):

<table>
<thead>
<th>Temperature</th>
<th>$p_{min}$</th>
<th>$\alpha$</th>
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
