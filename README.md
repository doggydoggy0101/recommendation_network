# Recommendation neural network

This is an implementation of personalized movie recommendation using a neural network model proposed in the paper: Deep Neural Networks for Youtube Recommendations.

## Introduction

A recommendation system like an information filtering system that will predict the preferences or ratings of a user would give to an item. For example, video recommendations are widely used within streaming services like Netflix, YouTube and so on. Imagine you open YouTube, but you have no idea what to watch. Now the youtube homepage’s recommendation comes to work. The more preferred videos they recommend to you, the more likely you will be using youtube, which is a great news for them.

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/sample.png" width="625" height="300">

This table shows a relation between 4 users and 5 movies. The first user watched, the second, third and fifth movies, and gives them a rating respectively. Note that because it is impossible for a user to watch every single movie, there will be a lot of unknown spaces in a real world situation. Our goal is to predict these unknown spaces using neural network.

### Model

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/model.png" width="625" height="300">

The overall structure of our recommendation network consists of two models, candidate generation and ranking. The candidate generation network generates a hundred movies from a large corpus. And then the ranking network ranks the a hundred movies and recommend maybe the top 10 movies to the user.

### Dataset

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/data.png" width="625" height="200">

https://grouplens.org/datasets/movielens/

https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

We use MovieLens data from kaggle, which contains 10 thousand users with 100 thousands of ratings corresponded to 45 thousands of movies. Also this data contains about 30 different features that we can explore through it.

## Candidate generation

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/candidate.png" width="625" height="300">

Basiclly, we just input the watching history, predict the top 100 possible future watches, only by the pattern of watching history. If you think about it, this is a very hard task to do, it is simply just impossible to predict what a person will watch in the future, so the accuracy in this stage will not be high. In order to recommend better movies, we need to rank them later.

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/candidate_1.png" width="625" height="225">

In the paper of youtube recommendation, adding depths to network improves the performance, and indeed it improves the learning speed. Here is the measure method we previously mentioned, we call it recall. By looking at the recall, it turns out that adding depths does not improve in our network. Again, high accuracy may not be a good sign in this stage, the data is easily overfitted in our case.

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/candidate_2.png" width="625" height="225">

Wider dense layer does improves the performance, we add more units until the convergence became difficult. Also we found that proper normalization helps either.

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/candidate_3.png" width="625" height="275">

With these parameters we get the best recall in about 20 epochs, and the mean of the testing data’s recall could go up to 41 percent.

## Ranking

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/ranking.png" width="625" height="300">

So we successfully generated 100 possible future watches, now we need to rank them. The ranking model has almost the same structure, although we add a lot more features to feed more user’s information, which makes the ranking network more personalized. We assign the best rated movie as label and predict the ratings in order to rank them.

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/ranking_1.png" width="625" height="275">

And again, we have to define a method to compare our ranking network. For this example, “the movie “Joe versus the vocano” is not in the right position if we assume the ranking be descending. We defined the error as the sum of misordered movies divided by the length of the matched movies. The minimum error occurs when we use one ReLU layer with 64 dense units.

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/ranking_2.png" width="625" height="250">

We test the network by adding and removing different features, popularity is a useful feature for ranking. As you can see, training without popularity does not perform well. The average rating of all the users actually doesn’t affect the network much. How a user rates a movie is influenced more by the movies popularity, but not the ratings by other users. Eventually, we try to train the network only with watching history and popularity, the performance is still quite well, and with much lesser training time.

## Results

<img src="https://github.com/doggydoggy0101/recommendation_network/blob/main/image/result.png" width="625" height="300">

At last, here is an example of a recommend. We split a testing data’s watch history into two part, feed the first part to the candidate generation network and predicts 100 possible watches, which is the left part here. We compare the 100 movies with the second part of the watching history, there are four movies matched here, and labled the ratings on it. Then the ranking network ranks the 100 movies for the user, predicts the best rated movies and recommend to the user.

## Reference

[Deep Neural Networks for Youtube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190) 
