# CMPUT466-Project
This is an email spam filter. 
To tackle the spam classification problem, I used a supervised machine learning approach
using logistic regression as the core algorithm. To further compare the accuracy of the chosen
approach, I used two additional machine learning algorithms: Naive Bayes and Random
Forest. Additionally, I implemented baseline models to evaluate the performance of our
approach and set benchmarks.


First, you run the train.py; and this will generate logistic_regression_weights.npy, logistic_regression_bias.npy, word2idx.json and idf.npy.
Then you can call the functions in predict.py to filter spam email
