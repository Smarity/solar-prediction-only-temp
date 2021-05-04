import pickle
import pandas as pd
import numpy as np
import StatsFunctions as stats
import hpelm
from sklearn.preprocessing import StandardScaler

class alm04_svm():
    """
    This class represents the best model/configuration from Tabernas (RIA station).
    
    It uses a ELM model and the following input configuration (being rs the predicted value):
        ['tx', 'tn', 'ra', 'delta_t', 'energyt', 'hormin_tx', 'tx_prev', 'rs']

    """
    def __init__(self):
        # import model
        filename = "models/elm_alm04_Tx_Tn_Ra_deltaT_EnergyT_HorminTx_Tn_prev_Rs.pkl"
        with open(filename, 'rb') as file:
            self.model =hpelm.ELM(inputs=7, outputs=1 )
            self.model.load(filename)

        # define required inputs
        self.parameters = ['tx', 'tn', 'ra', 'delta_t', 'energyt', 'hormin_tx', 'tx_prev', 'rs']

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
        mean_list = [23.21, 9.84, 29.28, 13.37, 364.57, 13.39, 23.21]
        std_list = [7.34, 6.17, 9.40, 3.87, 162.81, 1.89, 7.33]

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
    mlModel = alm04_svm()
    mlModel.import_dataset("data/data-daily-asheville-example.csv", 'csv')
    mlModel.getStandardDataTest()
    mlModel.predictValues()
    rmse, rrmse, mbe, r2, nse = mlModel.statAnalysis()
    print(rmse, rrmse, mbe, r2, nse)
