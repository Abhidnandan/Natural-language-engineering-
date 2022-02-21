# Natural-language-engineering-
# University of Essex
# School of Computer Science and Electronic Engineering
# CE887-7-AU : Natural Language Engineering
# Assignment:Build a text classifier on the IMDB sentiment classification dataset, you can use any classification method, but you must training your model on the first 40000 instances and testing your model on the last 10000 instances. The IMDB dataset will be uploaded on the moodle page for you to download.

Text classification is a machine learning technique that assigns a set of predefined categories to open-ended text. Text classifiers can be used to structure and categorize any kind of text from files and documents. Text classification is the automatic process of predicting one or more categories given a piece of text.
# Your code should include:
# 1: Read the file, incorporate the instances into the training set and testing set.
# 2: Pre-processing the text, you can choose whether you need stemming, removing stop words,removing non-alphabetical words. (Not all classification models need this step, it is OK if you think your model can perform better without this step, and you can give some justification in the report.)
# 3: Analysing the feature of the training set, report the linguistic features of the training dataset.
# 4: Build a text classification model, train your model on the training set and test your model on the test set.
# 5: Summarize the performance of your model.

***Import nltk, Import random to help us with the data set to shuffle so that we can get it random. After that, we assign the data set and test set.***

***From nltk. Corpus and import movie_reviews helps to import the data sets and reviews that can be positive as well as negative.***
 
 ***Import movie reviews as mr***
 
 ***Import string*** 
 
 ***From nltk corpus import stopwords***
 
 ***Define a variable stop and pass stopwords through it***
 
 ***Defining documents in lower case and removing punctuations as well as stopwords from the original document.***
 
 ***The data set I used to do sentiment analysis is IMDB Dataset, which contains the reviews and sentiment as features. The review features explain the review of the movie and the    sentiment gives the values of the movie review that’s is either a positive or negative.***
 
 ***d_file.IsNull().sum() this line we will be using to check for the missing values in the data set. And to check the value counts in the data sets we will use the     d_file[‘review’].value_counts() as a comment.***
 
 # Splitting the data set in the model 
***To measure the performance and to stop the overfitting of the model we need to split the data into training and testing sets.***

***From sklearn.model_selection import train _test_split  from the sklearn library and assign the review to the X parameter and assign sentiment to the y parameter and after that will continue doing the train test split for X and y by giving the test size 0.2 then we will get the first 40000 instances for training and last 10000 instances for testing in the model***
# Training the model
***we divided our data into training and testing sets, later by using a support vector machine we train the model and see how its works.
Later we will use the LinearSVC classifier from the sklearn.SVM library to train our model from the support vector machine algorithm.
From sklearn.pipeline import pipeline to build the pipeline to vectorize the data and then train and fit the model.
Then create a  t_clf object with our pipeline as a list of tuples or text and pass the ‘find’ instances to the Tfidfvectorizer and in the next line pass ‘clf’ instances to the Linear svc.
Now its time to fit the pipeline model X_train and y_train by commenting t_clf.fit(X_train,y_train).***
 # Evaluating the model   
***To predict and evaluate the model performance we use matrics such as confusion matrix, precision,f1 score, recall, and accuracy.
A confusion matrix is used to evaluate the performance of the classification model and it compares the values predicted by the model and it gives the performance of our classifier. In our model the confusion matrix is predicted as true positive for the sentiment is 4453 and the error is 537 whereas the negative for the negative sentiment is predicted as 4551 and the error is 459 times.    
Now run predictions by giving predictions =t_clf.predict(X_test) will predict on our test      set and analyze the result			.
Now report the confusion matrix from sklearn. metrics  and import confusion_matrix, classification_report and accuracy_score.***
# Print the overall accuracy 
***The accuracy gives the values of correct predictions and in our case, the overall accuracy of the model is predicted as 0.90%.***




 
 
 
 
