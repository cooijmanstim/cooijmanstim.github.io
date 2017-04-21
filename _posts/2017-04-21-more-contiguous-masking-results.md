---
layout: default
title: "More contiguous masking: results"
date:   2017-04-21
---

### More contiguous masking: results

Results after training the model with the masking strategy described in [the previous post](https://cooijmanstim.github.io/2017/04/17/more-contiguous-masking/).
In particular, I mask out 17x17 rectangles. This is squared EMD as before.

Things are getting much more interesting:

<table>
<thead>
<th>Strategy</th>
<th>Samples</th>
</thead>
<tbody>

<tr><td>Greedy</td><td>
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/9.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/10.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/17.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/0.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/19.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/7.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/6.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/1.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/18.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/16.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/8.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/11.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/2.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/5.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/12.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/15.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/14.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/13.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/4.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_greedy_ancestral_2017-04-19T20:34:08.217198_T1.0/3.gif">
</td></tr>

<tr><td>Antigreedy</td><td>
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/7.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/0.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/9.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/18.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/16.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/11.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/10.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/17.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/19.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/8.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/1.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/6.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/14.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/13.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/5.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/2.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/3.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/4.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/12.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_antigreedy_ancestral_2017-04-19T23:39:53.919793_T1.0/15.gif">
</td></tr>

<tr><td>Orderless</td><td>
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/11.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/16.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/18.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/0.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/7.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/9.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/19.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/17.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/10.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/8.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/6.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/1.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/2.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/5.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/13.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/14.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/4.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/3.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/15.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_orderless_ancestral_2017-04-19T18:14:27.027297_T1.0/12.gif">
</td></tr>

<tr><td>Gibbs</td><td>
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/5.gif ">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/2.gif ">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/17.gif ">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/10.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/19.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/3.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/4.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/18.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/11.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/16.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/15.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/12.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/9.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/7.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/0.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/13.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/14.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/1.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/6.gif">
<img src="assets/images/sample_2017-04-19/sample_emd2_deepish_contiguish_independent_gibbs_2017-04-19T15:55:25.238319_T1.0/8.gif">
</td></tr>

</tbody>
</table>

It's definitely looking like I need to tweak the Gibbs parameters. I'll also do another hyperparameter search to make sure I've got the right architecture.