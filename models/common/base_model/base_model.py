import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

class BaseModel():
    def fit(self):
        for emotion in self.emotions:
            self.all_predictions[emotion].fit(self.X, self.y[emotion])
            print('trained model for {}'.format(emotion.replace('_x', '')))

    def predict_emotions(self, text_arr) -> dict:
        predictions = {}
        
        for emotion in self.emotions:
            predictions[emotion] = self.all_predictions[emotion].predict(text_arr)
         
        return predictions

    def calculate_model_accuracy(self, x_test, y_test: pd.DataFrame):
        accuracy_sum = 0
        for emotion in self.emotions:
            predictions = self.all_predictions[emotion].predict(x_test)
            accuracy = accuracy_score(y_test[emotion], predictions) * 100
            accuracy_sum += accuracy
            print("Accuracy of model for emotion {} is {:.2f}%".format(emotion.replace('_x', ''), accuracy))

        print('Average accuracy score: {:.2f}'.format(accuracy_sum / len(self.emotions)))

    def _get_all_emotion_prediction_models(self):
        print(self.all_predictions)

    def get_best_params(self):
        for emotion in self.emotions:
            print('Best params for {} are {}'.format(emotion.replace('_x', ''),
                                                     self.all_predictions[emotion].best_params_))
            
    def get_confusion_matrix(self, x_train, y_true: pd.DataFrame, reduce_features=False):        
        y_pred = self.predict_emotions(x_train)
        
        if not reduce_features:
            for emotion in self.emotions:
                print(f'Confusion matrix for {emotion}\n{confusion_matrix(y_true[emotion].to_numpy(), np.array(y_pred[emotion]))}')
                
            for emotion in self.emotions:
                self._get_classification_report(y_true[emotion], y_pred[emotion], emotion)
        else:
            y_pred = pd.DataFrame(y_pred)
            y_pred.mask(y_pred > 0 , 1, inplace=True)
            y_pred.mask(y_pred < 0 , -1, inplace=True)
            y_pred.mask(y_pred == 0 , 0, inplace=True)
            
            y_true.mask(y_true > 0 , 1, inplace=True)
            y_true.mask(y_true < 0 , -1, inplace=True)
            y_true.mask(y_true == 0 , 0, inplace=True)
                        
            for emotion in self.emotions:
                print(f'Confusion matrix for {emotion}\n{confusion_matrix(y_true[emotion].to_numpy(), y_pred[emotion].to_numpy())}')
            
            for emotion in self.emotions:
                self._get_classification_report(y_true[emotion], y_pred[emotion], emotion)
    
    def _get_classification_report(self, y_true, y_pred, emotion):
        print(f'Classification report for {emotion}:\n{classification_report(y_true, y_pred)}')