import os
import pandas as pd
from models.common.data_preparation import create_vectorizer_and_prediction_features
from models.SVM_classification import SVMMultipleOutputClassification
from sklearn.model_selection import train_test_split

path = os.path.join('data', 'full_data','data.xlsx')
# path = os.path.join('data', 'sample_data','train.xlsx')


df_train = pd.read_excel(path, index_col=False)
Text_X_Tfidf, y_train = create_vectorizer_and_prediction_features(df_train)

x_train, x_test, y_train, y_test = train_test_split(Text_X_Tfidf, y_train, test_size=0.2)

def run_svm():
    clf = SVMMultipleOutputClassification(x_train, y_train)
    clf._get_all_emotion_prediction_models()

    clf.fit()
    clf.get_best_params()

    clf.calculate_model_accuracy(x_test, y_test)
    clf.get_confusion_matrix(x_test, y_test)
    clf.get_confusion_matrix(x_test, y_test, reduce_features=True)
  

if __name__ == '__main__':
    run_svm()

