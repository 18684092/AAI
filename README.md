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
P(W='strong'|PT='yes') = 3/9 = 0.333
```

When predicting...

```$ python "NaiveBayes.py" play_tennis-test.csv PT True```



```
P(PT='no'|evidence) = P(PT='no') * P(O='sunny'|PT='no') * P(T='cool'|PT='no') * P(H='high'|PT='no') * P(W='strong'|PT='no')
P(PT='yes'|evidence) = P(PT='yes') * P(O='sunny'|PT='yes') * P(T='cool'|PT='yes') * P(H='high'|PT='yes') * P(W='strong'|PT='yes')

P(PT='no'|evidence) = 0.357 * 0.6 * 0.2 * 0.8 * 0.6 =  0.021
P(PT='yes'|evidence) = 0.643 * 0.222 * 0.333 * 0.333 * 0.333 =  0.005

P(PT='no'|evidence) = 0.795
P(PT='yes'|evidence) = 0.205
```


or features can be binary...


```
P(Smoking='0') = 363/1500 = 0.242
P(Smoking='1') = 1137/1500 = 0.758
P(Yellow_Fingers='0') = 310/1500 = 0.207
P(Yellow_Fingers='1') = 1190/1500 = 0.793
P(Anxiety='1') = 938/1500 = 0.625
P(Anxiety='0') = 562/1500 = 0.375
P(Peer_Pressure='0') = 971/1500 = 0.647
P(Peer_Pressure='1') = 529/1500 = 0.353
P(Genetics='0') = 1281/1500 = 0.854
P(Genetics='1') = 219/1500 = 0.146
P(Attention_Disorder='1') = 471/1500 = 0.314
P(Attention_Disorder='0') = 1029/1500 = 0.686
P(Born_an_Even_Day='0') = 750/1500 = 0.5
P(Born_an_Even_Day='1') = 750/1500 = 0.5
P(Car_Accident='1') = 1081/1500 = 0.721
P(Car_Accident='0') = 419/1500 = 0.279
P(Fatigue='0') = 400/1500 = 0.267
P(Fatigue='1') = 1100/1500 = 0.733
P(Allergy='1') = 501/1500 = 0.334
P(Allergy='0') = 999/1500 = 0.666
P(Coughing='0') = 454/1500 = 0.303
P(Coughing='1') = 1046/1500 = 0.697
P(Lung_cancer='0') = 414/1500 = 0.276
P(Lung_cancer='1') = 1086/1500 = 0.724
P(Smoking='0'|Lung_cancer='0') = 240/414 = 0.58
P(Smoking='0'|Lung_cancer='1') = 123/1086 = 0.113
P(Smoking='1'|Lung_cancer='0') = 174/414 = 0.42
P(Smoking='1'|Lung_cancer='1') = 963/1086 = 0.887
P(Yellow_Fingers='0'|Lung_cancer='0') = 188/414 = 0.454
P(Yellow_Fingers='0'|Lung_cancer='1') = 122/1086 = 0.112
P(Yellow_Fingers='1'|Lung_cancer='0') = 226/414 = 0.546
P(Yellow_Fingers='1'|Lung_cancer='1') = 964/1086 = 0.888
P(Anxiety='1'|Lung_cancer='0') = 202/414 = 0.488
P(Anxiety='1'|Lung_cancer='1') = 736/1086 = 0.678
P(Anxiety='0'|Lung_cancer='0') = 212/414 = 0.512
P(Anxiety='0'|Lung_cancer='1') = 350/1086 = 0.322
P(Peer_Pressure='0'|Lung_cancer='0') = 286/414 = 0.691
P(Peer_Pressure='0'|Lung_cancer='1') = 685/1086 = 0.631
P(Peer_Pressure='1'|Lung_cancer='0') = 128/414 = 0.309
P(Peer_Pressure='1'|Lung_cancer='1') = 401/1086 = 0.369
P(Genetics='0'|Lung_cancer='0') = 408/414 = 0.986
P(Genetics='0'|Lung_cancer='1') = 873/1086 = 0.804
P(Genetics='1'|Lung_cancer='0') = 6/414 = 0.014
P(Genetics='1'|Lung_cancer='1') = 213/1086 = 0.196
P(Attention_Disorder='1'|Lung_cancer='0') = 111/414 = 0.268
P(Attention_Disorder='1'|Lung_cancer='1') = 360/1086 = 0.331
P(Attention_Disorder='0'|Lung_cancer='0') = 303/414 = 0.732
P(Attention_Disorder='0'|Lung_cancer='1') = 726/1086 = 0.669
P(Born_an_Even_Day='0'|Lung_cancer='0') = 201/414 = 0.486
P(Born_an_Even_Day='0'|Lung_cancer='1') = 549/1086 = 0.506
P(Born_an_Even_Day='1'|Lung_cancer='0') = 213/414 = 0.514
P(Born_an_Even_Day='1'|Lung_cancer='1') = 537/1086 = 0.494
P(Car_Accident='1'|Lung_cancer='0') = 239/414 = 0.577
P(Car_Accident='1'|Lung_cancer='1') = 842/1086 = 0.775
P(Car_Accident='0'|Lung_cancer='0') = 175/414 = 0.423
P(Car_Accident='0'|Lung_cancer='1') = 244/1086 = 0.225
P(Fatigue='0'|Lung_cancer='0') = 224/414 = 0.541
P(Fatigue='0'|Lung_cancer='1') = 176/1086 = 0.162
P(Fatigue='1'|Lung_cancer='0') = 190/414 = 0.459
P(Fatigue='1'|Lung_cancer='1') = 910/1086 = 0.838
P(Allergy='1'|Lung_cancer='0') = 147/414 = 0.355
P(Allergy='1'|Lung_cancer='1') = 354/1086 = 0.326
P(Allergy='0'|Lung_cancer='0') = 267/414 = 0.645
P(Allergy='0'|Lung_cancer='1') = 732/1086 = 0.674
P(Coughing='0'|Lung_cancer='0') = 283/414 = 0.684
P(Coughing='0'|Lung_cancer='1') = 171/1086 = 0.157
P(Coughing='1'|Lung_cancer='0') = 131/414 = 0.316
P(Coughing='1'|Lung_cancer='1') = 915/1086 = 0.843
```