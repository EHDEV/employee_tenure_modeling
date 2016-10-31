import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

APPLICANTS_SCHEMA = {"user_id": "int64",
                     "client_name": "str",
                     "answer1": "int64",
                     "answer2": "int64",
                     "answer3": "int64",
                     "answer4": "int64",
                     "answer5": "int64",
                     "answer6": "int64",
                     "answer7": "int64",
                     "answer8": "int64",
                     "answer9": "int64",
                     "answer10": "int64",
                     "answer11": "int64",
                     "answer12": "int64",
                     "answer13": "int64",
                     "answer14": "int64",
                     "answer15": "int64",
                     "answer16": "int64",
                     "answer17": "int64",
                     "answer18": "int64",
                     "answer19": "int64",
                     "answer20": "int64",
                     "answer21": "int64",
                     "answer22": "int64",
                     "answer23": "int64",
                     "answer24": "int64",
                     "answer25": "int64",
                     "log_total_time": "float64",
                     "device": "str"}

HIRES_SCHEMA = {"user_id": "int64",
                "client": "str",
                "tenure_length": "int64",
                "currently_employed": "str",
                "hire_job_category": "str"}


def dummiefy(var):
    dummified = pd.get_dummies(var)
    dummified.columns = [
        var.name + "_" + str(int(c)) if type(c) != str else var.name + "_" + str(c) for c in dummified.columns
        ]
    return dummified


def encode_target(target):
    le = LabelEncoder()
    le.fit(target)
    return le.transform(target), list(le.classes_)


def impute_nan(col):
    """
    imputes nans with -1 or empty string
    :param col:
    :return:
    """
    if col.name == 'device':
        col = col.astype(str)
        col[col.str.lower().str.contains('samsung')] = 'samsung'
        col[col.str.upper().str.contains('LG')] = 'LG'
        col[col.str.upper().str.contains('HTC')] = 'HTC'

    if col.dtype in ['str', 'O'] or col.name.find('answer') >= 0:
        return col.astype('category')
    elif col.dtype.name == 'category':
        return col
    else:
        return col.fillna(-1)


class DataLoadTransform(object):
    """
    Data Extraction, loading and munging are performed for this project using this class.
    """

    def __init__(self, data_dir=None):
        """
        Construct the paths to the data sources and load the raw datasets into pandas DataFrames
        """
        load_dotenv(find_dotenv())
        self.data_dir = data_dir or os.path.realpath(os.environ.get('DATA_DIR'))
        applicants_file = os.path.join(self.data_dir, os.environ.get('APPLICANTS_DATA'))
        hires_file = os.path.join(self.data_dir, os.environ.get('HIRES_DATA'))
        self.hires = pd.read_csv(hires_file, error_bad_lines=False,
                                 skipinitialspace=True)
        self.applicants = pd.read_csv(applicants_file, error_bad_lines=False,
                                      skipinitialspace=True)

    @staticmethod
    def group_tenure_length(x):
        """
        Buckets teunre_length into three groups
        (less than 6 months, 6 - 12 months, or greater than 12 months)
        - For simplicity in calculation, it is assumed that the number of days in a month is 30
        :param x: int. tenure length of employee in days
        :return:
        """
        if np.isnan(x):
            pass
        if round(x / 30) < 6:
            return '0 to 6 Months'
        elif round(x / 30) < 12:
            return '6 to 12 Months'
        else:
            return 'More than 12 Months'

    def init_load_data(self):
        """
        Merges loan and institution DataFrames and returns a left merged dataframe [by As_of_Year, Respondent_ID
        and Agency_Code] that has an additional column (Respondent_Name_TS)
        :return: pandas DataFrame
        """
        expanded_df = self.applicants.merge(
            self.hires, how='inner', on=['user_id']
        )
        expanded_df = expanded_df[np.isnan(expanded_df.tenure_length) == False]
        expanded_df['tenure_length_category'] = expanded_df.tenure_length.apply(DataLoadTransform.group_tenure_length)
        return expanded_df

    @staticmethod
    def split_test_train(X, y, test_size=0.2):
        """
        Randomly split the data into test and train sets
        :param data:
        :param test_percentage:
        :return: list of indices for the test set
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        return X_train, y_train, X_test, y_test


    @staticmethod
    def prepare_data_for_modeling(data):
        """
        Transform variables appropriately and return a multidimensional array X, target variable y,
        and target class description
        :param data: DataFrame
        :return: tuple (numpy multidimensional array, numpy array, list)
        """
        exclude_columns = ['client', 'client_name', 'user_id', 'tenure_length', 'tenure_length_category']
        include_columns = data.columns.difference(exclude_columns)
        processed_df = pd.DataFrame(columns=[])
        for col in include_columns:
            data[col] = impute_nan(data[col])
            if data[col].dtype.name == 'category':
                processed_df = pd.concat((processed_df, dummiefy(data[col])), axis=1)
            else:
                processed_df = pd.concat((processed_df, data[col]), axis=1)
        y, class_description = encode_target(data.tenure_length_category)
        X = processed_df.values
        return X, y, class_description


if __name__ == '__main__':
    dlt = DataLoadTransform()
    raw_data = dlt.init_load_data()
    X, y, class_desc = DataLoadTransform.prepare_data_for_modeling(raw_data)
    X_train, y_train, X_test, y_test = DataLoadTransform.split_test_train(X, y)
    print("Train X and y: ", X_train.shape, y_train.shape)
    print("Test X and y: ", X_test, y_test)