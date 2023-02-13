import pandas as pd
from nltk.corpus import stopwords
import nltk
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer


stop = stopwords.words('english')

def basic_clean(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in basic_clean(x) if word not in (stop)]))

    return df

def create_vectorizer_and_prediction_features(df: pd.DataFrame) -> tuple:
    df = prepare_data(df)
    
    Tfidf_train_vect_train = TfidfVectorizer(max_features=10000)
    Tfidf_train_vect_train.fit(df['text'])
    Text_X_Tfidf = Tfidf_train_vect_train.transform(df['text'])
    y = df.iloc[:, 1:]
    
    return Text_X_Tfidf, y