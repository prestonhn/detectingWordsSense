# Detecting Word Sense
Program that utilizes machine learning models to predict word senses for given sentences that are inputted by the user. Models are trained on training data. Uses NLTK and sckit-learn.

I am choosing three words "rubbish", "yarn" and "tissue." Each of these words have different meanings or "sense" depending on how it is used in a sentence. We are training model on 2 different senses of each word and after training, the code will detect sense of word.

# Running Project
for training model: "python code/train_data.py"
for testing model: "python code/test_data.py"


# Data Collection and Sample Process 
collected 50 sentences for word "rubbish"
sense 1: rubbish, trash, scrap (worthless material that is to be disposed of)
sense 2: folderol, rubbish, tripe, trumpery, trash, wish-wash, applesauce, codswallop (nonsensical talk or writing)
25 sentences for each sense

Similarly collected 50 sentences for word "yarn"
sense 1: narration, recital, yarn (the act of giving an account describing incidents or a course of events)
sense 2: thread, yarn (a fine cord of twisted fibers (of cotton or silk or wool or nylon etc.) used in sewing and weaving)
25 sentences for each sense

Similarly collected 50 sentences for word "tissue"
sense 1: tissue (part of an organism consisting of an aggregate of cells having a similar structure and function)
sense 2: tissue, tissue paper (a soft thin (usually translucent) paper)
25 sentences for each sense

## Data source: 
"https://www.collinsdictionary.com/sentences/english/"

# Project Details
## Inputs
train data inputs:
rubbish.csv (this file contains inputs of rubbish word sentences on which we are training models)
tissue.csv (this file contains inputs of tissue word sentences on which we are training models)
yarn.csv (this file contains inputs of yarn word sentences on which we are training models)

test data inputs:
rubbish.csv (this file contains inputs of rubbish word sentences on which we are testing models)
tissue.csv (this file contains inputs of tissue word sentences on which we are testing models)
yarn.csv (this file contains inputs of yarn word sentences on which we are testing models)

## Outputs
train data output:
when we train models on different sentences of rubbish , yarn and tissue there will be 3 output files
rubbish.pkl
tissue.pkl
yarn.pkl

## Test Data Output
rubbish.csv (this file contains outputs of rubbish word sentences in first column and second column will be sense of that word)
tissue.csv (this file contains outputs of tissue word sentences in first column and second column will be sense of that word)
yarn.csv (this file contains outputs of yarn word sentences in first column and second column will be sense of that word)

## Project Display
I am using a csv file for saving results 1 column has sentences and other column has sense of that words.



