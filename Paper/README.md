Counting the Cockroaches in the Walls or I Know Why the Caged Bird
Tweets: Assessing the Severity of Service Failures Through Social
Chatter
================

## Abstract

Firms employ a variety of passive listening platforms (e.g., customer
comment cards, help lines, social media) that elicit a variety of
complaints. Since these complaints are self-selected, it isn’t obvious
if complaints are widespread or genuinely indicative of systemic service
failures. In other words, how can can a firm know when complaints can be
ignored and when they need to addressed? This issue is related to a
larger class of problems where the goal is to make inference about the
size of a population from a non-random sample (e.g., the German tank
problem). In this paper, we develop a model that uses a variety of
observed customer complaints on Twitter to make inference about the
severity of service failures.

## Introduction

It has been said that if you observe a cockroach on your floor, there
are likely thousands inside the walls of your home. In a similar
fashion, a variety of marketing studies have been conducted (by both
academics and practitioners) that attempt to provide similar estimates
with respect to the severity of service failures. For example, a
standard rule of thumb is only 10% of service failures are reported.
Firms need information on the severity (i.e., type and reach) of service
failures in order to know how and when to correct such failures. Our
interest in this paper is to develop a model that uses readily available
data from social media channels to estimate the severity of service
failures.

Using complaints (i.e, reports of service failures) on social media to
estimate the type and reach of service failures is an opportunity and a
challenge. The opportunity is access to a much larger, current, and
inexpensive dataset compared to the traditional ways of assessing
service failure severity (e.g., complaint cards and customer surveys).
The challenge is considerable, including the non-representativeness of
the complaints (i.e., not everyone uses social media and not everyone
who experiences a complaint reports it) and that those using social
media may be incentivized, given the public nature of the network, to
exaggerate complaints.

To illustrate, suppose a firm uses feedback reported on Twitter to
assess service failure severity in order to simply determine their
competitive position in the marketplace (e.g., the firm with the fewest
complaints is the market leader). To do this, they simply count the
number of negative tweets (i.e., complaints) for each firm for a given
type of service failure. After accounting for differences in firm size,
how accurately do these counts measure the true size of the population
affected by the service failure (i.e., the reach of the service
failure)? The counts could  the population size because not everyone
that experiences a failure tweets about it. However, users could tweet
more than once, experience multiple failures, tweet a complaint without
experiencing the failure, or be influenced by network effects – which
would all lead to observed counts that  the true population size. We
propose a model that explicitly disentangles the propensity to complain
from the propensity to exaggerate complaints.

## Model Specification

### Classifying Complaints and Complaint Types

### Estimating Population Size

Service failure severity of a given type is synonymous with the  or  of
the service failure (i.e., the proportion of consumers experiencing the
service failure). However, since service failure complaints are
self-reported, we also need to model the probability of observing a
complaint (i.e., the propensity to complain). This first component is
analogous to models of animal abundance and occurrence in ecology (see
Royle and Dorazio, 2006; Wu et al., 2015). Additionally, since there are
incentives to complain on social media, we also need to model the
probability of exaggerating complaints (i.e., the propensity to
exaggerate). This second component represents a departure from the
ecology model and will be highlighted in .

Let’s focus on Twitter data associated with the airline industry. First,
let’s specify the baseline model without the propensity to exaggerate
(i.e., the ecology model). Second, let’s specify the alternative model
that accounts for the propensity to exaggerate.

% Baseline Model

% Alternative Model

where

% Ecological Model Examples

% Covariates and Data Sources

## Empirical Application

## Results

## Discussion

## References

Moderators of why people share word-of-mouth (Berger, 2014)
