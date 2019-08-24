
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import re
from os import path
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier



def predict(test_data, countVectorModel, tfidfModel, mlModel, outputExlFile):
    #LOAD MODEL
    print('Entering predict')
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open(countVectorModel,'rb')))
    loaded_tfidf = pickle.load(open(tfidfModel, 'rb'))
    loaded_model = pickle.load(open(mlModel, 'rb'))

    df_new = pd.read_excel(test_data)

    new_doc = df_new['STORY']
    mytxt_clean = new_doc.apply(text_cleaner)
    X_new_counts = loaded_vec.transform(mytxt_clean)
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)

    predicted_df = pd.DataFrame(data=predicted, columns=['SECTION'])

    predicted_df.to_excel(outputExlFile, index=False)
    
    print('Exiting predict')


def text_cleaner(document):
    stemmer = WordNetLemmatizer()

    document = re.sub(r'\W', ' ', str(document))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    return document

def countVectorizer(documents, countVectorModel):
    print('Entering countVectorizer')

    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(documents)
    
    pickle.dump(count_vect.vocabulary_, open(countVectorModel,'wb'))
    
    print('Exiting countVectorizer')

    return counts

def tfifTransformer(counts, tfidfModel):
    print('Entering tfifTransformer')

    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(counts)
    
    pickle.dump(tfidf_transformer, open(tfidfModel, 'wb'))
    
    print('Exiting tfifTransformer')

    return tfidf

def generate_model(train_data, countVectorModel, tfidfModel, svmModelFile):
    print('Entering generate_model')

    df = pd.read_excel(train_data)
    X = df.loc[:,'STORY']
    Y = df.loc[:,'SECTION']

    X_clean = X.apply(text_cleaner)
    X_counts = countVectorizer(X_clean, countVectorModel)
    X_tfidf = tfifTransformer(X_counts, tfidfModel)


    classifier = SGDClassifier().fit(X_tfidf, Y)

    '''
    y_pred = classifier.predict(X_test)


    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    '''

    #save model
    pickle.dump(classifier, open(svmModelFile, 'wb'))
    
    print('Exiting generate_model')


def main():
    train_data = 'data/Data_Train.xlsx'
    test_data = 'data/Data_Test.xlsx'
    pred_data = 'data/submission.xlsx'

    countVectorModel = 'model/count_vector.pkl'
    tfidfModel = 'model/tfidf.pkl'
    mlModel = 'model/sgd.pkl'

    if not (path.exists(countVectorModel) and 
        path.exists(tfidfModel) and 
        path.exists(mlModel)):
        generate_model(train_data, countVectorModel, tfidfModel, mlModel)
    
    predict(test_data, countVectorModel,tfidfModel, mlModel, pred_data)

if __name__ == "__main__":
    main()