# Advanced Artificial Intelligence

## Workshop 1 - NaiveBayes

Implement own Naive Bayes in Python. Calculates discrete probabilities for all features, then calculates the probability of each feature given another features, **P(feature='option'|given='option') = fraction = probability**.

So far it learns:

```$ python "NaiveBayes.py" play_tennis-train.csv PT```

```
P(O='sunny') = 5/14 = 0.357
P(O='overcast') = 4/14 = 0.286
P(O='rain') = 5/14 = 0.357
P(T='hot') = 4/14 = 0.286
P(T='mild') = 6/14 = 0.429
P(T='cool') = 4/14 = 0.286
P(H='high') = 7/14 = 0.5
P(H='normal') = 7/14 = 0.5
P(W='weak') = 8/14 = 0.571
P(W='strong') = 6/14 = 0.429
P(PT='no') = 5/14 = 0.357
P(PT='yes') = 9/14 = 0.643
P(O='sunny'|PT='no') = 3/5 = 0.6
P(O='sunny'|PT='yes') = 2/9 = 0.222
P(O='overcast'|PT='no') = 0/5 = 0.0
P(O='overcast'|PT='yes') = 4/9 = 0.444
P(O='rain'|PT='no') = 2/5 = 0.4
P(O='rain'|PT='yes') = 3/9 = 0.333
P(T='hot'|PT='no') = 2/5 = 0.4
P(T='hot'|PT='yes') = 2/9 = 0.222
P(T='mild'|PT='no') = 2/5 = 0.4
P(T='mild'|PT='yes') = 4/9 = 0.444
P(T='cool'|PT='no') = 1/5 = 0.2
P(T='cool'|PT='yes') = 3/9 = 0.333
P(H='high'|PT='no') = 4/5 = 0.8
P(H='high'|PT='yes') = 3/9 = 0.333
P(H='normal'|PT='no') = 1/5 = 0.2
P(H='normal'|PT='yes') = 6/9 = 0.667
P(W='weak'|PT='no') = 2/5 = 0.4
P(W='weak'|PT='yes') = 6/9 = 0.667
P(W='strong'|PT='no') = 3/5 = 0.6
P(W='strong'|PT='yes') = 3/9 = 0.333```