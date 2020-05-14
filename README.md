## About this project
This project is to implement and demonstrate the effectiveness of a classifier that first scales feature vectors by a naive bayes log count ratio before using a linear classifier (SVM was chosen here but others ie logistic regression can also be used).  This classifier, NBSVM, comes from a paper [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018.pdf) and [repo](https://github.com/sidaw/nbsvm)  by Sida Wang and Christopher D. Manning. 

Although it is widely acknowledged in research, (over 900 citations), it's not too well known amongst data science practitioners. FastAI's creator Jeremy Howard has wrote about it on [Kaggle](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline) but there was a time bottleneck in his implementation. This implementation has fixed the bottleneck (from >1hr to a few seconds at most) as well as using an Linear SVM classifier for speed. 

We will be comparing NBSVM to both bidirectional LSTM's and CNN's to assess it's performance. 

## Paper Details

The NBSVM approach is to weigh each feature by a "log-count ratio" before feeding the new weighted features into a linear classifier like SVM or logistic regression.

As per the paper, the ith training vector x(i) has shape (1xm) where m is the number of features in our text corpus. 

We then create 2 new vectors **p** and **q** both the size (1 x m). The vector **p** is created by adding all vectors x(i) where its associated label is 1. Likewise, **q** is created by adding all vectors where its associated label is -1 (or 0 or whatever numbers you use to represent classes).  Note that we add 1 to each element in vectors **p** and **q**. This helps work around both divide by 0 and log0 issues (an example of additive smoothing). 

The log count ratio **r**, is simply the logarithm of the normalized vector **p** divided by the normalized vector **q**.

Finally, we weigh each training vector x(i) by multiplying each element of x(i) with each element of **r** (element wise multiplication). And so we are ready to feed into a linear classifier!

## Results
|Model| Test Accuracy |
|--|--|
| NBSVM |  91.52%|
| LSTM|  88.85%|
| CNN|  85.14%|

We were able to confirm in our own implementation that NBSVM achieves an accuracy of around 91%, and better than 2 neural net architectures. Note that the [state of the art approaches](https://paperswithcode.com/sota/sentiment-analysis-on-imdb)  uses some variations of naive bayes weighted embeddings, LSTM's, CNN's or Transformers ( ie BERT).
