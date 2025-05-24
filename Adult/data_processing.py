import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import datasets


def data_processing():
    ### The function encoder the dataset, split the dataset to two coomunities
    # data upload
    MyDataFrame = pd.read_csv('adult.csv')
    # print(MyDataFrame.shape)
    MyDataFrame.columns = ['Age', 'Work Class', 'Final Weight', 'Education', 'Education Number', 'Marital Status',
                           'Occupation', 'Relationship', 'Race', 'Gender', 'Capital Gain', 'Capital Loss',
                           'Hours Per Week', 'Native Country', 'Salary']

    # data repair
    MyDataFrame['Occupation'] = MyDataFrame['Occupation'].replace('?', 'Prof-specialty')
    MyDataFrame['Work Class'] = MyDataFrame['Work Class'].replace('?', 'Private')
    MyDataFrame['Native Country'] = MyDataFrame['Native Country'].replace('?', 'United-States')

    MyDataFrame['Education'].value_counts()
    # data split to numerical and categorical
    numerical_columns = MyDataFrame[
        ['Age', 'Final Weight', 'Education Number', 'Capital Gain', 'Capital Loss', 'Hours Per Week']]
    categorical_columns = MyDataFrame[['Education', 'Work Class', 'Marital Status', 'Occupation', 'Relationship',
                                       'Race', 'Native Country', 'Gender']]
    sensitive_attribute_colums = MyDataFrame['Gender']
    community_label = MyDataFrame['Education']
    label_columns = MyDataFrame[['Salary']]

    # data encode

    # numerical encode
    numerical_ = numerical_columns.values
    norm = MinMaxScaler().fit(numerical_)
    numerical = norm.transform(numerical_)

    # catergorical encode
    encoder = OneHotEncoder(sparse_output=False)
    categorical = encoder.fit_transform(categorical_columns)

    # label encode
    y = label_columns.values
    lab = LabelEncoder()
    y = lab.fit_transform(y)
    y = np.reshape(y, (len(y), 1))
    print(np.shape(y))

    # sensitive attribute encode
    sensitive_attribute = sensitive_attribute_colums.values
    sensitive_attribute = lab.fit_transform(sensitive_attribute)
    sensitive_attribute = np.reshape(sensitive_attribute, (len(sensitive_attribute), 1))
    print(np.shape(sensitive_attribute))

    # train_test split
    dataset = np.concatenate((numerical, categorical, y,sensitive_attribute), axis=-1)
    # print(np.shape(dataset))
    # np.random.seed(1337)
    np.random.shuffle(dataset)


    # split data to different communities, one is phd, one is non-phd
    # community 1 is non_phd
    index1 = np.where(dataset[:, 16] == 0)
    dataset_community_1 = dataset[index1]
    np.random.shuffle(dataset_community_1)

    # community 2 is phd
    index2 = np.where(dataset[:, 16] == 1)
    dataset_community_2 = dataset[index2]
    # np.random.seed(1337)
    np.random.shuffle(dataset_community_2)

    print(' The data shape of community 1 is:'+str(dataset_community_1.shape)+' The data shape of community 2 is:'+str(dataset_community_2.shape))
    # community_1
    client_1_features=dataset_community_1[:,:-2]
    client_1_labels=dataset_community_1[:,-2]
    client_sensitive_attribute_1=dataset_community_1[:,-1]
    # community_2
    client_2_features=dataset_community_2[:,:-2]
    client_2_labels=dataset_community_2[:,-2]
    client_sensitive_attribute_2=dataset_community_2[:,-1]
    # train_vali_test split
    train_features_1, test_features_1, train_labels_1, test_labels_1, train_sensitive_1, test_sensitive_1 = train_test_split(client_1_features, client_1_labels,client_sensitive_attribute_1, test_size=0.2, random_state=42)
    train_features_2, test_features_2, train_labels_2, test_labels_2, train_sensitive_2, test_sensitive_2 = train_test_split(client_2_features, client_2_labels,client_sensitive_attribute_2, test_size=0.2, random_state=42)

    train_features_1, vali_features_1, train_labels_1, vali_labels_1, train_sensitive_1, vali_sensitive_1 = train_test_split(train_features_1, train_labels_1,train_sensitive_1, test_size=0.4, random_state=42)
    train_features_2, vali_features_2, train_labels_2, vali_labels_2, train_sensitive_2, vali_sensitive_2 = train_test_split(train_features_2, train_labels_2,train_sensitive_2, test_size=0.4, random_state=42)
    print('The data shape of community 1 is:'+str(train_features_1.shape)+' The data shape of community 2 is:'+str(train_features_2.shape))
    print('The data shape of community 1 is:'+str(vali_features_1.shape)+' The data shape of community 2 is:'+str(vali_features_2.shape))
    print('The data shape of community 1 is:'+str(test_features_1.shape)+' The data shape of community 2 is:'+str(test_features_2.shape))
    return train_features_1, vali_features_1, test_features_1, train_labels_1, vali_labels_1, test_labels_1, train_sensitive_1, vali_sensitive_1, test_sensitive_1, train_features_2, vali_features_2, test_features_2, train_labels_2, vali_labels_2, test_labels_2, train_sensitive_2, vali_sensitive_2, test_sensitive_2
