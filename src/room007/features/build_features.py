#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from room007.models import word_tag_predictor
from room007.models import advanced_train_and_predict
from room007.data import info



args = advanced_train_and_predict.get_arguments()
train_dataframes, test_dataframes = advanced_train_and_predict._get_data(args)
data_info = info.FeaturedData(name="simple")
feature_creator = word_tag_predictor.Features()
feature_creator.save = True
for name, dataframe in train_dataframes.items():
    feature_creator.save_features_to_dataframe(dataframe)
info.save_training_data(data_info, train_dataframes)
for name, dataframe in test_dataframes.items():
    feature_creator.save_features_to_dataframe(dataframe)
info.save_test_data(data_info, test_dataframes)
