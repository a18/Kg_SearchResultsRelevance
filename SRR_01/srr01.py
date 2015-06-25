#!/usr/bin/env python

# *************************************** #
#  Search Results Relevance @ Kaggle
#  Author: https://www.kaggle.com/benhamner/crowdflower-search-relevance/python-benchmark
#  Date: 2015-06-24
#  Modified by: andy1618
#
# *************************************** #

import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import random
import time

# Column names
COL_query_tokens_in_title = 'query_tokens_in_title'
COL_query_tokens_in_description = 'query_tokens_in_description'
COL_query = 'query'
COL_product_title = 'product_title'
COL_product_description = 'product_description'

RND_SEED = 1
RF_TREES = 200
MIN_SAMPLES_SPLIT = 2

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b") # 2+ words
    data[COL_query_tokens_in_title] = 0.0
    data[COL_query_tokens_in_description] = 0.0
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row[COL_query]))
        title = set(x.lower() for x in token_pattern.findall(row[COL_product_title]))
        description = set(x.lower() for x in token_pattern.findall(row[COL_product_description]))
        if len(title) > 0:
            data.set_value(i, COL_query_tokens_in_title, len(query.intersection(title))/len(title))
        if len(description) > 0:
            data.set_value(i, COL_query_tokens_in_description, len(query.intersection(description))/len(description))

if __name__ == '__main__':

    start = time.time()
    random.seed(RND_SEED)

    #                          Feature Set Name            Data Frame Column                Transformer
    features = FeatureMapper([('QueryBagOfWords',          COL_query,                       CountVectorizer(max_features=200)),
                              ('TitleBagOfWords',          COL_product_title,               CountVectorizer(max_features=200)),
                              ('DescriptionBagOfWords',    COL_product_description,         CountVectorizer(max_features=200)),
                              ('QueryTokensInTitle',       COL_query_tokens_in_title,       SimpleTransform()),
                              ('QueryTokensInDescription', COL_query_tokens_in_description, SimpleTransform())])

    print "Reading files..."
    train = pd.read_csv("../input/train.csv").fillna("")
    test = pd.read_csv("../input/test.csv").fillna("")

    print "Extracting features..."
    extract_features(train)
    extract_features(test)

    pipeline = Pipeline([("extract_features", features),
                         ("classify", RandomForestClassifier(n_estimators=RF_TREES,
                                                             n_jobs=1,
                                                             min_samples_split=MIN_SAMPLES_SPLIT,
                                                             random_state=RND_SEED))])

    print "Fitting model..."
    pipeline.fit(train, train["median_relevance"])

    print "Predicting ..."
    predictions = pipeline.predict(test)

    print "Writing out data ..."
    submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
    submission.to_csv("_out/python_benchmark.csv", index=False)

    print "Done. Elapsed:", time.time() - start
