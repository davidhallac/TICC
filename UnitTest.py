import unittest
import TICC_solver as TICC
import numpy as np
import sys




class TestStringMethods(unittest.TestCase):

    def test_example(self):
        fname = "example_data.txt"
        (cluster_assignment, cluster_MRFs) = TICC.solve(window_size = 1,number_of_clusters = 8, lambda_parameter = 11e-2, beta = 600, maxIters = 100, threshold = 2e-5, write_out_file = False, input_file = fname, prefix_string = "output_folder/", num_proc=1)
        assign = np.loadtxt("UnitTest_Data/Results.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        for i in range(8):
            mrf = np.loadtxt("UnitTest_Data/cluster_"+str(i)+".txt",delimiter=',')
            try:
                np.testing.assert_array_almost_equal(mrf, cluster_MRFs[i])
            except AssertionError:
                #Test failed
                self.assertTrue(1==0)             


    def test_multiExample(self):
        fname = "example_data.txt"
        (cluster_assignment, cluster_MRFs) = TICC.solve(window_size = 5,number_of_clusters = 5, lambda_parameter = 11e-2, beta = 600, maxIters = 100, threshold = 2e-5, write_out_file = False, input_file = fname, prefix_string = "output_folder/", num_proc=1)
        
        assign = np.loadtxt("UnitTest_Data/multiResults.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        for i in range(5):
            mrf = np.loadtxt("UnitTest_Data/multiCluster_"+str(i)+".txt",delimiter=',')
            try:
                np.testing.assert_array_almost_equal(mrf, cluster_MRFs[i])
            except AssertionError:
                #Test failed
                self.assertTrue(1==0)


if __name__ == '__main__':
    unittest.main()


