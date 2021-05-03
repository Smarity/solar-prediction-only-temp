import unittest
from cor06_svm import cor06_svm
import os

# go to root location
os.chdir("..")
pathTest = os.getcwd()

class TestCor06(unittest.TestCase):
    def test_importModel(self):
        mlModel = cor06_svm()
        self.assertTrue(mlModel)

    def test_importDataset(self):
        mlModel = cor06_svm()
        mlModel.import_dataset("data/data-daily-ashaville-example.csv", 'csv')
        self.assertNotEqual(mlModel.dfData.shape[0], 0, 'Error importing file')

    def test_parameters(self):
        mlModel = cor06_svm()
        mlModel.import_dataset("data/data-daily-ashaville-example.csv", 'csv')
        len_parameters = len(mlModel.parameters)
        self.assertEqual(mlModel.dfData.shape[1], len_parameters)



if __name__ == '__main__':
    unittest.main()
