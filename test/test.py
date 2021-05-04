import unittest
import os
from cor06_svm import cor06_svm

# go to root location
os.chdir("..")
pathTest = os.getcwd()

class TestCor06(unittest.TestCase):
    def test_importModel(self):
        mlModel = cor06_svm()
        self.assertTrue(mlModel)

    def test_importDataset(self):
        """
        Check we correctly import data
        """
        mlModel = cor06_svm()
        mlModel.import_dataset("data/data-daily-asheville-example.csv", 'csv')

        self.assertNotEqual(
            first=mlModel.dfData.shape[0], second=0,
            msg='Error importing file')

    def test_parameters(self):
        """
        Check the dataset contains all parameters
        """
        mlModel = cor06_svm()
        mlModel.import_dataset("data/data-daily-asheville-example.csv", 'csv')
        len_parameters = len(mlModel.parameters)

        self.assertEqual(
            first=mlModel.dfData.shape[1], second=len_parameters,
            msg="Error filtering the parameters from dataset")

    def test_splitDataset(self):
        mlModel = cor06_svm()
        mlModel.import_dataset("data/data-daily-asheville-example.csv", 'csv')
        x_test, y_test = mlModel.getStandardDataTest()

        self.assertEqual(first=x_test.shape[0], second=y_test.shape[0])

    def test_predictionValues(self):
        mlModel = cor06_svm()
        mlModel.import_dataset("data/data-daily-asheville-example.csv", 'csv')
        mlModel.getStandardDataTest()
        mlModel.predictValues()

        self.assertEqual( mlModel.y_pred.shape[0], mlModel.x_test.shape[0])

if __name__ == '__main__':
    unittest.main()
