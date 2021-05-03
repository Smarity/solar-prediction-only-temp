import pickle
import pandas as pd
from sklearn.svm import SVR

class cor06_svm():
    """
    This class represents the best model/configuration from Córdoba (RIA station).
    
    It uses a SVM model and the following input configuration (being rs the predicted value):
        ['tx', 'tn', 'ra', 'deltat', 'energyt', 'hormintx', 'tx_prev', 'tn_next', 'rs']

    """

    def __init__(self):
        # import model
        filename = "models/svm_cor06_Tx_Tn_Ra_deltaT_EnergyT_HorminTx_Tx_prev_Tn_next_Rs.sav"
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)

        # define required inputs
        self.parameters = ['tx', 'tn', 'ra', 'deltaT', 'EnergyT', 'HorminTx', 'Tx_prev', 'Tn_next', 'Rs']

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

        # filter by parameters
        #self.dfData = self.dfData.filter(self.parameters)

    def predict(self):
        pass

if __name__ == '__main__':
    mlModel = cor06_svm()
    mlModel.import_dataset("data/data-daily-ashaville-example.csv", 'csv')
    print(mlModel.dfData.info())
