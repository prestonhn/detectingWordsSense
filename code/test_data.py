from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import nltk
import joblib
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    lemmas = [wn.morphy(token) or token for token in tokens]
    lemmas = ' '.join(lemmas)
    return lemmas

def wsd_test(sentence, model_file):
    cv, clf = joblib.load(model_file)
    X_test = cv.transform([preprocess(sentence)])
    return clf.predict(X_test)[0] + 1

def WSD_Test_Rubbish(sentences):
    model_file = 'model/rubbish.pkl'
    results = {}
    for sentence in sentences:
        result = wsd_test(sentence, model_file)
        results[sentence] = result
    return results

def WSD_Test_Yarn(sentences):
    model_file = 'model/yarn.pkl'
    results = {}
    for sentence in sentences:
        result = wsd_test(sentence, model_file)
        results[sentence] = result
    return results

def WSD_Test_Tissue(sentences):
    model_file = 'model/tissue.pkl'
    results = {}
    for sentence in sentences:
        result = wsd_test(sentence, model_file)
        results[sentence] = result
    return results

def read_test_data(test_filename):
    df = pd.read_csv(test_filename, delimiter=';', header=0)
    return df['sentences'].to_list()

def print_results(results):
    for key,value in results.items():
        print(f"\"{key}\" : sense {value}")

def save_results(results,word):
    columns = ['sentence', 'sense']
    df = pd.DataFrame({col: [] for col in columns})
    if(word == "rubbish"):
        senses_list = []
        for val in results.values():
            if(val == 1):
                senses_list.append("rubbish, trash, scrap (worthless material that is to be disposed of)")
            else:
                senses_list.append("folderol, rubbish, tripe, trumpery, trash, wish-wash, applesauce, codswallop (nonsensical talk or writing)")
        df['sentence'] = results.keys()
        df['sense'] = senses_list
        df.to_csv('results/rubbish.csv', index=False)
    if(word == "yarn"):
        senses_list = []
        for val in results.values():
            if(val == 1):
                senses_list.append("narration, recital, yarn (the act of giving an account describing incidents or a course of events) \"his narration was hesitant\"")
            else:
                senses_list.append("thread, yarn (a fine cord of twisted fibers (of cotton or silk or wool or nylon etc.) used in sewing and weaving)")
        df['sentence'] = results.keys()
        df['sense'] = senses_list
        df.to_csv('results/yarn.csv', index=False)
    if(word == "tissue"):
        senses_list = []
        for val in results.values():
            if(val == 1):
                senses_list.append("tissue (part of an organism consisting of an aggregate of cells having a similar structure and function)")
            else:
                senses_list.append("tissue, tissue paper (a soft thin (usually translucent) paper)")
        df['sentence'] = results.keys()
        df['sense'] = senses_list
        df.to_csv('results/tissue.csv', index=False)




rubbish_sentences = read_test_data("data/test_data/rubbish.csv")
yarn_sentences = read_test_data("data/test_data/yarn.csv")
tissue_sentences = read_test_data("data/test_data/tissue.csv")
print("** Testing Rubbish **")
print("1: (n) rubbish, trash, scrap (worthless material that is to be disposed of) \n2: (n) folderol, rubbish, tripe, trumpery, trash, wish-wash, applesauce, codswallop (nonsensical talk or writing) \n")
print("** Output Results for word Rubbish **")
rubbish_results = WSD_Test_Rubbish(rubbish_sentences)
print_results(rubbish_results)
save_results(rubbish_results,"rubbish")
print("")
print("** Testing Yarn **")
print("1: (n) narration, recital, yarn (the act of giving an account describing incidents or a course of events) \"his narration was hesitant\" \n2: (n) thread, yarn (a fine cord of twisted fibers (of cotton or silk or wool or nylon etc.) used in sewing and weaving) \n")
print("** Output Results for word Yarn **")
yarn_results = WSD_Test_Yarn(yarn_sentences)
print_results(yarn_results)
save_results(yarn_results,"yarn")
print("")
print("1: (n) tissue (part of an organism consisting of an aggregate of cells having a similar structure and function) \n2: tissue, tissue paper (a soft thin (usually translucent) paper) \n")
print("** Output Results for word Tissue **")
print("Testing Tissue")
tissue_results = WSD_Test_Tissue(tissue_sentences)
print_results(tissue_results)
save_results(tissue_results,"tissue")