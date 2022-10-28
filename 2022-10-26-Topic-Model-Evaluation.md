---
layout: post
title: Evaluating Topic Models - Why it's Hard
date: 2022-10-26
categories: [topic modeling, non-technical]
---

**This post carries the following assumptions**
1. You are familiar with topic modeling (specifically with LDA)
2. You are familiar with the metrics mentioned here

As such, I will **not** be going over them except in the most cursory way. Indeed,
the main focus here is why this is a hard task. I'll follow up with evaluating in practice.

# Topic Model Evaluation (optimal K)
This is a topic I found ridiculously difficult to find information on. Whether
in the literature, or online. On a conceptual level, this isn't terribly surprising.
When you ask how to find the optimal number of topics, of course it will be difficult.
**If you spot a glaring error, please feel free to email me about it and I will fix it.
Please forgive me if it seems as if I'm pontificating here; it's something I'm
passionate about.**

As a thought exercise, let's think about the following sentences:
> "John went to the State Fair with Wenya. While there they rode rides, ate food,
played games, and checked out the booths. They had such a great time, they decided they should go again next
year."

How many topics would you say there are? That's not an easy question from a human
standpoint. You could come up with a number of different topics...and while they
aren't all equally valid (let's say the topic of Politics for instance), there
isn't just one answer. In general, if we as humans struggle with something with
consistency, it will prove difficult to do that same thing with a computer.

Let's think about how LDA works for a second. LDA assumes that documents are a
mixture of topic probabilities. Or in human terms, any document *could* be a number
of topics. That makes things a bit messy for us. While the most probable label
will end up applied, there are still likely to be many labels vying for the honor
of being applied to a given document. This is important for us because, it is this
uncertainty that arguable reflects the same human issues here. If you run LDA
5 times, you will be technically getting back 5 different models.  

Let's think of those models as different people. Different people with different
ways of seeing the world. They have different boundaries for deciding what topics
exist. So these different models (like people) will give different labels. This
means that over a corpus of documents, they may well have different ideas of the
number of topics. Does our problem begin to get clearer?

Of course, as part of LDA, we set the number of topics at the start. This forces
the application of the best label (even if it's terrible). We can set and test
a number of K values and see how they turn out according to a number of metrics.
Now, as someone working with topic modeling, I would ***love*** if we could have
a nice clean answer to how to find optimal K. Unfortunately, we can't.

This is because there isn't exactly just one optimal K value (bear with me here).
If you think in terms of finding just one value, you're going to struggle. That's
why I prefer to think of it as a matter of finding several different "Keighborhoods"
(K neighborhoods; pardon my puniness).

Let's pivot away from topic modeling for a moment and think of library catalogues.
Categorizing books by topic is quite complicated. That's why most classification
schemes will have subcategories. So let's say you have a copy of Murach's MySQL,
now under the Library of Congress system, it has a classification of QA76.9.D32 M846 2015.
The Q means it falls under science, the QA means it falls under Mathematics, the
76 puts it under instruments and machines.
For more, see the classification outline [here](https://www.loc.gov/catdir/cpso/lcco/).

This little aside, serves to illustrate that classification can be (and often is)
hierarchical in the real world. Thus, topics can have subtopics. It all depends
on the level of granularity you're looking for. Thus, while LDA doesn't have
subtopics built in, by increasing K, you are increasing the granularity of the
topics.

Why does this matter? It's important because of the idea of multiple values of K
that *could* apply. Because there are a couple of K value ranges which make sense
(because) of the subtopics that can occur, you don't have just one area to look
at. In theory, if you wanted to, you could probably go to many thousands of topics
and be super granular. And that would be totally fine if your goal is to have a
machine use the labels in some way. However, if you want to have people understand
and interpret those...it becomes problematic.

# (Internal) Metrics
Let's think about metrics and evaluation for a minute. Our goal with topic
modeling is to try to represent a corpus of documents in a richer way than mere
term frequency. Thus, ideally, our topic models should represent the corpus
accurately ***and*** be solidly human-interpretable. There are a number of different
metrics we can look at for this but they all measure different things. Some of them
are actually *negatively* correlated with human-interpretability. That means we
can have a perfectly good topic model (from the metrics) with topics that seem
like garbage to a human.

This is why one of the most common pieces of advice you will see online is to
look at each of the topics generated to see if it makes sense. And I tend to agree,
the most important metric is expert human opinion. However, this isn't workable
at scale. That probably explains why so many research papers I read tried only
a few (or even only 1) value(s) for K. That unwieldiness at scale is why I think
metrics have a place. Even with their difficulties.

One of the concepts you'll hear bandied about is the elbow method (or gap statistic).
This is considered to be broadly applicable across clustering techniques (which
topic modeling could be considered). I'm not entirely sure that it has had a
robust analysis of its effectiveness conducted so I tend to shy away from it.
Further, there are few implementations within Python of it and they carry...their
own issues. Thus, I am not overly fond of it.

If we look at other metrics, we want to focus on Internal Metrics. This is because
if we don't know the ground truth of topics (and most of the time we won't) these
are important. These metrics can include Silhouette Coefficients, Calinski-Harabasz,
among others. I think [Lossio-Ventura et al. 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9040385/pdf/nihms-1704563.pdf) provides a decent coverage of some of these (mainly Silhouette Coefficients
and Calinski-Harabasz). [Rudiger et al. 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9049322/)
provide an excellent overview of topic modeling metrics such as perplexity and
coherence. Of course, these internal metrics don't necessarily lead to the best
human-interpretable topics as they are focused on the goodness of the clusters.
Indeed, Rudiger et al. 2022 note the research showing that perplexity doesn't correlate
with human interpretability. These metrics are not all equally effective across
all methods. Further, the data itself can impact the effectiveness of the metric.
This demands a solid understanding of what metric to use where, which is something
 hard to find in the literature in my experience. Thus, my preference is to use
 multiple metrics, try a number of K values, and find Keighborhoods to investigate
 further by looking at the topics formed. This should be clearer with my follow-up
 post where I'll show how I've done it in the past.
