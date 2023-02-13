import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from models.common.base_model.base_model import BaseRegressionModel

class SVMMultipleOutputClassification(BaseRegressionModel):
    def __init__(self, X, y: pd.DataFrame):
        self.X = X
        self.y = y
        self.parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
        self.emotions = y.columns
        self.all_predictions = {emotion: GridSearchCV(svm.SVC(), self.parameters) for emotion in self.emotions}
        