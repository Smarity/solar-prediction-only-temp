import pickle
import pandas as pd
import numpy as np
import StatsFunctions as stats
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class ash08_xgb():
    """
    This class represents the best model/configuration from Asheville (NCEI station) - aridity index = 1.1494.
    
    It uses a XGB model and the following input configuration (being rs the predicted value):
        ['tx', 'tn', 'ra', 'deltat', 'energyt', 'hormin_tx', 'tx_prev', 'rs']

    """
    def __init__(self):
        # import model
        filename = "models/xgb_ash08_tx_tn_ra_deltaT_energyt_hormin_tx_tx_prev_rs.json"
        self.model = xgb.XGBRegressor()
        self.model.load_model(filename)

        # define required inputs
        self.parameters = ['tx', 'tn', 'ra', 'deltat', 'energyt', 'hormin_tx', 'tx_prev', 'rs']

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

        #define deltat
        self.dfData["deltat"] = self.dfData['tx'] - self.dfData['tn']
        # convert rs from W/m2 to MJ/m2day-1
        self.dfData["rs"] = self.dfData['rs']*0.0864
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
        mean_list = [20.30, 6.67, 30.38, 13.62, 209.17, 14.49, 20.14]
        std_list = [7.86, 8.31, 9.04, 4.93, 1033.36, 2.25, 8.03]

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
        self.y_pred = np.array(self.model.predict(self.x_test))
        return self.y_pred

    def statAnalysis(self):
        rmse = stats.get_root_mean_square_error(self.y_test, self.y_pred)
        rrmse = stats.get_root_mean_square_error(self.y_test, self.y_pred) / np.mean(self.y_test)
        mbe = stats.get_mean_bias_error(self.y_test, self.y_pred)
        r2 = stats.get_coefficient_of_determination(self.y_test, self.y_pred)
        nse = stats.get_nash_suteliffe_efficiency(self.y_test, self.y_pred)

        return rmse, rrmse, mbe, r2, nse

if __name__ == '__main__':
    mlModel = ash08_xgb()
    mlModel.import_dataset("data/ncei-asheville-example.csv", 'csv')
    mlModel.getStandardDataTest()
    mlModel.predictValues()
    rmse, rrmse, mbe, r2, nse = mlModel.statAnalysis()
    print(rmse, rrmse, mbe, r2, nse)