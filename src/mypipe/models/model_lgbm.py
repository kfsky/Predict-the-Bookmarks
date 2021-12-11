from lightgbm import LGBMModel
from lightgbm import LGBMClassifier
from .base import BaseModel
from sklearn.metrics import f1_score
import numpy as np


class MyLGBMModel(BaseModel):
    def build_model(self):
        model = LGBMModel(**self.params)
        return model

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.model = self.build_model()
        self.model.fit(tr_x, tr_y,
                       eval_set=[[va_x, va_y]],
                       **self.fit_params)


class MyLGBMClassifier(BaseModel):
    def build_model(self):
        model = LGBMClassifier(**self.params)
        return model

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.model = self.build_model()
        self.model.fit(tr_x, tr_y,
                       eval_set=[[va_x, va_y]],
                       **self.fit_params)
