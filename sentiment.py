#import necessary libraries
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import joblib
import random


#download the NLTK data files (run this once)
nltk.download("movie_reviews")
nltk.download("punkt")
nltk.download("stopwords")


#Data preprocessing
def extract_features(words):
    return {word: True for word in words}


#Load the movie reviews dataset from nltk
documents=[(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

#shuffle the dataset to ensure random distribution and selection 
random.shuffle(documents)

#prepare the dataset for training and testing
featuresets=[(extract_features(d), c) for (d, c) in documents]
train_set, test_set=featuresets[:1600], featuresets[1600:]


#train the Naive Bayes classifier
classifier=NaiveBayesClassifier.train(train_set)


#Evaluate the classifier on the test set
accuracy=nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%") #print the accuracy of the model to 2 decimal places


#show the most informative features
classifier.show_most_informative_features(10)


#test on new input sentences
def analyze_sentiment(text):
    #tokenize and remove stopwords
    words=nltk.word_tokenize(text)
    words=[word for word in words if word.lower() not in stopwords.words("english")]
    
    #predict sentiment
    features=extract_features(words)
    return classifier.classify(features)

#example test sentences to test the classifier
test_sentences=[
    "This movie is absolutely fantastic! The acting, the story, everything was amazing!",
    "I hated this movie. It was a waste of time and money.",
    "The plot was a bit dull, but the performances were great.",
    "I have mixed feelings about this film. It was okay, not great but not terrible either."
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentiment(sentence)}")


#save the trained model for future use
joblib.dump(classifier, "sentiment_model.joblib")
