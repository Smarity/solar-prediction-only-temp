import pickle
import pandas as pd
import numpy as np
import StatsFunctions as stats
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class gra03_svm():
    """
    This class represents the best model/configuration from Loja (RIA station).
    
    It uses a SVM model and the following input configuration (being rs the predicted value):
        ['tx', 'tn', 'ra', 'delta_t', 'energyt', 'hormin_tx', 'tn_prev', 'rs']

    """
    def __init__(self):
        # import model
        filename = "models/mlp_gra03_Tx_Tn_Ra_deltaT_EnergyT_HorminTx_Tn_prev_Rs.h5"
        self.model = keras.models.load_model(filename)

        # define required inputs
        self.parameters = ['tx', 'tn', 'ra', 'delta_t', 'energyt', 'hormin_tx', 'tn_prev', 'rs']

    def import_dataset(self, fileLocation, fileType):
        """
        This function import a dataset and convert it into a pandas Dataframe
        Inputs:
            fileLocation: "data/dataSet.csv"
            fileType: string with csv, excel or txt
        """
        if fileType == 'csv' or fileType == 'txt':
            self.dfData = pd.read_csv(fileLocation)
        elif fileType == 'excel':
            self.dfData = pd.read_excel(fileLocation)
        else:
            AssertionError('this fileType does not exit, use excel, csv and txt instead')

        # define deltat
        self.dfData["deltat"] = self.dfData['tx'] - self.dfData['tn']
        # convert rs from W/m2 to MJ/m2day-1
        self.dfData["rs"] = self.dfData['rs'] * 0.0864
        # filter nan values
        self.dfData.dropna(inplace=True)
        self.dfData.reset_index(drop=True, inplace=True)
        # filter the parameters
        self.dfData = self.dfData.filter(self.parameters)

    def getStandardDataTest(self):
        """
        Split data to train and test
        """
        # from training original dataset
        mean_list = [23.26, 9.33, 29.02, 13.93, 370.37, 14.21, 9.33]
        std_list = [8.52, 6.22, 9.42, 4.71, 177.41, 2.15, 6.23]

        # we have the input data as x, and the output as y
        self.x_test = np.array(self.dfData.iloc[:, :-1])
        self.y_test = np.array(self.dfData.iloc[:, -1])

        # standarization
        scaler = StandardScaler()
        scaler.mean_ = mean_list
        scaler.scale_ = std_list

        # x_train and x_test
        self.x_test = np.array(scaler.transform(self.x_test))

        return self.x_test, self.y_test

    def predictValues(self):
        self.y_pred = np.ravel(np.array(self.model.predict(self.x_test)))
        return self.y_pred

    def statAnalysis(self):
        rmse = stats.get_root_mean_square_error(self.y_test, self.y_pred)
        rrmse = stats.get_root_mean_square_error(self.y_test, self.y_pred) / np.mean(self.y_test)
        mbe = stats.get_mean_bias_error(self.y_test, self.y_pred)
        r2 = stats.get_coefficient_of_determination(self.y_test, self.y_pred)
        nse = stats.get_nash_suteliffe_efficiency(self.y_test, self.y_pred)

        return rmse, rrmse, mbe, r2, nse

if __name__ == '__main__':
    mlModel = gra03_svm()
    mlModel.import_dataset("data/data-daily-asheville-example.csv", 'csv')
    mlModel.getStandardDataTest()
    mlModel.predictValues()
    rmse, rrmse, mbe, r2, nse = mlModel.statAnalysis()
    print(rmse, rrmse, mbe, r2, nse)