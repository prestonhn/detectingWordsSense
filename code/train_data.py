from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    lemmas = [wn.morphy(token) or token for token in tokens]
    lemmas = ' '.join(lemmas)
    return lemmas

def train_wsd(sense_file):
    sense1_sentences = []
    sense2_sentences = []
    df = pd.read_csv(sense_file, delimiter=';', header=0)
    sense1_sentences = df['sense1'].to_list()
    sense2_sentences = df['sense2'].to_list()
    cv = CountVectorizer()
    X = cv.fit_transform(sense1_sentences + sense2_sentences)
    Y = [0] * len(sense1_sentences) + [1] * len(sense2_sentences)

    clf = MultinomialNB()
    clf.fit(X, Y)
    model_file_name = sense_file.split("/")[2]
    model_file_name = model_file_name.split(".")[0]
    model_file = f'model/{model_file_name}.pkl'
    joblib.dump((cv, clf), model_file)


train_wsd("data/train_data/rubbish.csv")
train_wsd("data/train_data/yarn.csv")
train_wsd("data/train_data/tissue.csv")

