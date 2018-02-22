import pandas as pd
import numpy as np

class ExtractFeature:
    def __init__(self, column):
        self.column = column

    def transform(self, data):
        feature_data = data[self.column].values
        return [[x] for x in feature_data]

    def fit(self, *_):
        return self

class ExtractBinaryFeature:
    def __init__(self, column):
        self.column = column

    def transform(self, data):
        binary_data = data[self.column].apply(lambda y: 0 if pd.isnull(y) else 1).values
        return [[x] for x in binary_data]

    def fit(self, *_):
        return self

class ExtractLastOnlineFeature:
    def __init__(self, column):
        self.column = column
    
    def process_row(self, row):
        if isinstance(row.lastOnline, basestring):
            try:
                val = int(row.lastOnline.split()[0])
                return val
            except Exception:
                return np.nan 
  
        return row.lastOnline

    def transform(self, data):
        last_online_data = data.apply(self.process_row, axis='columns')
        return [[x] for x in last_online_data]

    def fit(self, *_):
        return self