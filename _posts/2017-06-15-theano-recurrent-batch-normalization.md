---
layout: default
title: Less painful Recurrent Batch Normalization in Theano
date:   2017-06-15
---

### Less painful Recurrent Batch Normalization in Theano
(Alternative title: the only good thing I ever done)

My debut [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025) works great but is impossible to implement without breaking absolutely all of your abstractions. Personally I've switched to [Layer Normalization](https://arxiv.org/abs/1607.06450), but thanks to fellow MILA student Olivier Mastropietro I was reminded of [a script](https://github.com/cooijmanstim/Attentive_reader/blob/bn/codes/att_reader/popstats.py) I wrote back when I was working on recurrent batch normalization.

In particular, I was shoehorning batch norm into Caglar Gulcehre's Attentive Reader code. As always when working with Other People's Code(tm), I try to avoid venturing into the bowels, so I wrote code that will take an arbitrary BN-enabled training graph and hack it up to get the corresponding inference graph.

The way it works is -- as you construct your training graph, you label all tensors `x` that represent batch statistics by setting `x.tag.bn_statistic = True`. Then you train your model. Once the model is trained, you call the function `get_inference_graph`, giving it your model's symbolic inputs, symbolic outputs, and an iterable over batches of training data. So `get_inference_graph` traverses your graph to find all batch statistics, creates a new graph that does the same thing but has all the batch statistics as additional outputs, and runs this graph on the provided training data to estimate population statistics. Then, it creates the inference graph, which is like the original graph but has batch statistics replaced by population statistics.

This would be trivial if it weren't for Theano's `scan` feature, because everything would be trivial if it weren't for Theano's `scan` feature. Scan encodes loops in the graph, and the way it does this is by representing its loop body as a completely separate "inner" graph, with "inner" inputs that correspond to "outer" inputs to the Scan node, and "inner" outputs that correspond to the "outer" outputs of the Scan node. They correspond but they are not connected, and so you can't just take batch statistics from the "inner" graph and connect them to stuff in the "outer" graph.

In order to export the batch statistics to the "outer" graph in order to estimate population statistics, my code has to add them as extra Scan outputs. Once I get the population statistics, I need to import them back into the "inner" graph by adding appropriate Scan inputs, i.e. `nitsots` and `mitmots` and `whatnots`.

In the end, the code works for any level of nesting of Scan loops (including when you have no Scan at all). I suppose it would have been strategic to publicize this code back then, but perhaps it can still serve as a historical testament to the wicked rituals we had to go through back in the day of symbolic computation graphs. You know, before we all switched to PyTorch.

The code is [here](https://github.com/cooijmanstim/Attentive_reader/blob/bn/codes/att_reader/popstats.py) and a usage example is [here](https://github.com/cooijmanstim/Attentive_reader/blob/bn/codes/att_reader/attentive_reader.py#L286-L346). I reproduce my baby it in all its glory below:

```python
import sys, cPickle as pkl
import theano, itertools, pprint, copy, numpy as np, theano.tensor as T
from collections import OrderedDict
from theano.gof.op import ops_with_inner_function
from theano.scan_module.scan_op import Scan
from theano.scan_module import scan_utils

def equizip(*sequences):
    sequences = list(map(list, sequences))
    assert all(len(sequence) == len(sequences[0]) for sequence in sequences[1:])
    return zip(*sequences)

# get outer versions of the given inner variables of a scan node
def export(node, extra_inner_outputs):
    assert isinstance(node.op, Scan)

    # this is ugly but we can't use scan_utils.scan_args because that
    # clones the inner graph and then extra_inner_outputs aren't in
    # there anymore
    old_inner_inputs = node.op.inputs
    old_inner_outputs = node.op.outputs
    old_outer_inputs = node.inputs

    new_inner_inputs = list(old_inner_inputs)
    new_inner_outputs = list(old_inner_outputs)
    new_outer_inputs = list(old_outer_inputs)
    new_info = copy.deepcopy(node.op.info)

    # put the new inner outputs in the right place in the output list and
    # update info
    new_info["n_nit_sot"] += len(extra_inner_outputs)
    yuck = len(old_inner_outputs) - new_info["n_shared_outs"]
    new_inner_outputs[yuck:yuck] = extra_inner_outputs

    # in step 8, theano.scan() adds an outer input (being the actual
    # number of steps) for each nitsot. we need to do the same thing.
    # note these don't come with corresponding inner inputs.
    offset = (1 + node.op.n_seqs + node.op.n_mit_mot + node.op.n_mit_sot +
              node.op.n_sit_sot + node.op.n_shared_outs)
    # the outer input is just the actual number of steps, which is
    # always available as the first outer input.
    new_outer_inputs[offset:offset] = [new_outer_inputs[0]] * len(extra_inner_outputs)

    new_op = Scan(new_inner_inputs, new_inner_outputs, new_info)
    outer_outputs = new_op(*new_outer_inputs)

    # grab the outputs we actually care about
    extra_outer_outputs = outer_outputs[yuck:yuck + len(extra_inner_outputs)]
    return extra_outer_outputs

def gather_symbatchstats_and_estimators(outputs):
    symbatchstats = []
    estimators = []
    visited_scan_ops = set()

    for var in theano.gof.graph.ancestors(outputs):
        if hasattr(var.tag, "bn_statistic"):
            var.tag.original_id = id(var)
            symbatchstats.append(var)
            estimators.append(var)

        # descend into Scan
        try:
            op = var.owner.op
        except:
            continue
        if isinstance(op, Scan) and op not in visited_scan_ops:
            visited_scan_ops.add(op)
            print "descending into", var

            inner_estimators, inner_symbatchstats = gather_symbatchstats_and_estimators(op.outputs)
            outer_estimators = export(var.owner, inner_estimators)

            symbatchstats.extend(inner_symbatchstats)
            estimators.extend(outer_estimators)

    return symbatchstats, estimators

def get_population_outputs(batch_outputs, popstats):
    replacements = []
    visited_scan_ops = set()

    for var in theano.gof.graph.ancestors(batch_outputs):
        if hasattr(var.tag, "bn_statistic"):
            # can't rely on object identity because scan_args clones; use original_id
            popstat = next(popstat for batchstat, popstat in popstats.items() if batchstat.tag.original_id == var.tag.original_id)
            replacements.append((var, T.patternbroadcast(popstat, var.broadcastable)))

        # descend into Scan
        try:
            op = var.owner.op
        except:
            continue
        if isinstance(op, Scan):
            # this would cause multiple replacements for this variable
            assert not hasattr(var.tag, "bn_statistic")

            if op in visited_scan_ops:
                continue
            visited_scan_ops.add(op)
            print "descending into", var

            node = var.owner
            sa = scan_utils.scan_args(outer_inputs=node.inputs, outer_outputs=node.outputs,
                                      _inner_inputs=node.op.inputs, _inner_outputs=node.op.outputs,
                                      info=node.op.info)

            # add subscript as sequence
            # TODO check if this integer input drops the scan to cpu, if so use float and cast back in subtensor expression
            indices = T.arange(sa.n_steps)
            index = scan_utils.safe_new(indices[0])
            sa.outer_in_seqs.append(indices)
            sa.inner_in_seqs.append(index)

            # add popstats as nonsequences (because they may be shorter than len(indices))
            inner_popstats = {}
            for batchstat, outer_popstat in popstats.items():
                # this can't be subscripted hence won't appear in the inner graph
                if outer_popstat.ndim == 0:
                    continue

                inner_popstat = scan_utils.safe_new(outer_popstat)
                sa.outer_in_non_seqs.append(outer_popstat)
                sa.inner_in_non_seqs.append(inner_popstat)

                inner_popstats[batchstat] = theano.ifelse.ifelse(index < inner_popstat.shape[0],
                                                                 inner_popstat[index],
                                                                 inner_popstat[-1])

            # recurse on inner graph
            new_inner_outputs = sa.inner_outputs
            new_inner_outputs = get_population_outputs(new_inner_outputs, inner_popstats)

            # construct new scan node
            new_op = Scan(sa.inner_inputs, new_inner_outputs, sa.info)
            new_outer_outputs = new_op(*sa.outer_inputs)

            # there is one-to-one correspondence between old outer
            # inputs and new_outer_inputs; replace one-to-one
            replacements.extend(equizip(node.outputs, new_outer_outputs))

    print "replacements", replacements
    population_outputs = scan_utils.clone(batch_outputs, replace=replacements)
    return population_outputs

def get_inference_graph(inputs, batch_outputs, estimation_batches):
    symbatchstats, estimators = gather_symbatchstats_and_estimators(batch_outputs)
    print "symbatchstats x estimators", equizip(symbatchstats, estimators)

    if not symbatchstats:
        print "NO BATCH STATISTICS FOUND IN GRAPH"
    #assert symbatchstats

    def aggregate_varlen(aggregate, sample):
        # grow to accomodate shape
        aggregate = np.pad(aggregate,
                           [(0, max(0, sample.shape[j] - aggregate.shape[j]))
                            for j in range(aggregate.ndim)],
                           mode="constant")
        aggregate[tuple(map(slice, sample.shape))] += sample
        return aggregate

    # take average of batch statistics over estimation_batches
    estimator_fn = theano.function(inputs, estimators, on_unused_input="warn")
    batchstats = {}
    for i, batch in enumerate(estimation_batches):
        estimates = estimator_fn(**batch)
        for symbatchstat, estimator, estimate in equizip(symbatchstats, estimators, estimates):
            batchstats.setdefault(symbatchstat, []).append(estimate)

    popstats = {}
    coverages = {}
    for symbatchstat in symbatchstats:
        if batchstats[symbatchstat][0].ndim > 1:
            # assume first axis is time
            maxlen = max(map(len, batchstats[symbatchstat]))
            # pad all batch stats to maxlen by repeating last time step
            padded_batchstats = [
                np.pad(batchstat,
                       [(0, maxlen - len(batchstat))] + [(0, 0) for _ in range(1, batchstat.ndim)],
                       mode="edge")
                for batchstat in batchstats[symbatchstat]]
            popstat = sum(bs / len(padded_batchstats) for bs in padded_batchstats)

            coverages[symbatchstat] = (
                np.arange(maxlen)[None, :] <
                np.asarray(list(map(len, batchstats[symbatchstat])))[:, None]
            ).sum(axis=0)
        else:
            # not time-separated, just average as is
            popstat = sum(bs / len(batchstats[symbatchstat]) for bs in batchstats[symbatchstat])
        popstats[symbatchstat] = popstat

    if True:
        # allow inspection of all_stats
        import matplotlib.pyplot as plt
        for symbatchstat, popstat in popstats.items():
            if popstat.ndim == 1:
                plt.figure()
                plt.hist(popstat)
                plt.title(symbatchstat.tag.bn_label)
            elif False:
                plt.matshow(popstat, cmap="bone")
                plt.colorbar()
            else:
                choice = np.random.choice(popstat.shape[1], size=(min(20, popstat.shape[1]),), replace=False)
                fig, axes = plt.subplots(2, sharex=True)

                axes[0].plot(popstat[:, choice])
                axes[0].set_title("values")

                axes[1].plot(coverages[symbatchstat])
                axes[1].set_title("support")
                axes[1].set_ylabel("batches")
                axes[1].set_xlabel("time steps")

                fig.suptitle(symbatchstat.tag.bn_label)

            plt.savefig("%s.pdf" % symbatchstat.tag.bn_label, bbox_inches="tight")
        #plt.show()
        #import pdb; pdb.set_trace()
        pkl.dump(dict(batchstats=batchstats, coverages=coverages, popstats=popstats),
                 open("allstats.pkl", "wb"))

    sympopstats = {}
    for symbatchstat, popstat in popstats.items():
        # need as_tensor_variable to make sure it's not a CudaNdarray
        # because then the replacement will fail as symbatchstat may not
        # have been moved to the gpu yet.
        sympopstat = T.as_tensor_variable(theano.shared(popstat)).copy(name="popstat_%s" % symbatchstat.name)
        sympopstats[symbatchstat] = sympopstat

    population_outputs = get_population_outputs(batch_outputs, sympopstats)

    return population_outputs
```