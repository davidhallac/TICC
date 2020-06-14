import unittest
from TICC_solver import TICC
import numpy as np
import sys


class TestStringMethods(unittest.TestCase):

    def test_example(self):
        fname = "example_data.txt"
        ticc = TICC(window_size = 1,number_of_clusters = 8, lambda_parameter = 11e-2, beta = 600, maxIters = 100,
                    threshold = 2e-5, write_out_file = False, prefix_string = "output_folder/", num_proc=1)
        (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)
        assign = np.loadtxt("UnitTest_Data/Results.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        # Test prediction works with batch of data outside of `fit` method. Perhaps there is a better way
        # to test this in parallel so these are more like unit tests rather than integration tests?
        test_batch = ticc.predict_clusters(ticc.trained_model['complete_D_train'][0:1000, ])
        batch_val = abs(test_batch - cluster_assignment[0:1000])
        self.assertEqual(sum(batch_val), 0)

        # Test streaming by passing in 5 row blocks at a time (current timestamp and previous 4)
        # I am causing data leakage by training on the whole set and then using the trained model while streaming,
        # but this is for testing the code, so it is ok
        # TODO: figure out why larger blocks don't improve predictions more. Reference:
        # https://github.com/davidhallac/TICC/issues/18#issuecomment-384514116
        def test_streaming(block_size):
            test_stream = np.zeros(1000)
            test_stream[0:block_size] = cluster_assignment[0:block_size]
            for i in range(block_size, 1000):
                point = ticc.trained_model['complete_D_train'][i - block_size:i, ]
                test_stream[i] = ticc.predict_clusters(point)[block_size - 1]

            percent_correct_streaming = 100 * sum(cluster_assignment[0:1000] == test_stream) / 1000.0
            self.assertGreater(percent_correct_streaming, 0.9)

        test_streaming(5)

        for i in range(8):
            mrf = np.loadtxt("UnitTest_Data/cluster_"+str(i)+".txt",delimiter=',')
            try:
                np.testing.assert_array_almost_equal(mrf, cluster_MRFs[i], decimal=3)
            except AssertionError:
                #Test failed
                self.assertTrue(1==0)


    def test_multiExample(self):
        fname = "example_data.txt"
        ticc = TICC(window_size = 5,number_of_clusters = 5, lambda_parameter = 11e-2, beta = 600, maxIters = 100,
                    threshold = 2e-5, write_out_file = False, prefix_string = "output_folder/", num_proc=1)
        (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)
        assign = np.loadtxt("UnitTest_Data/multiResults.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        for i in range(5):
            mrf = np.loadtxt("UnitTest_Data/multiCluster_"+str(i)+".txt",delimiter=',')
            try:
                np.testing.assert_array_almost_equal(mrf, cluster_MRFs[i], decimal=3)
            except AssertionError:
                #Test failed
                self.assertTrue(1==0)

    def test_biased_vs_unbiased(self):
        fname = "example_data.txt"
        unbiased_ticc = TICC(window_size=1, number_of_clusters=8, lambda_parameter=11e-2, beta=600, maxIters=100,
                             threshold=2e-5,
                             write_out_file=False, prefix_string="output_folder/", num_proc=1)
        (unbiased_cluster_assignment, unbiased_cluster_MRFs) = unbiased_ticc.fit(input_file=fname)

        biased_ticc = TICC(window_size=1, number_of_clusters=8, lambda_parameter=11e-2, beta=600, maxIters=100,
                           threshold=2e-5,
                           write_out_file=False, prefix_string="output_folder/", num_proc=1, biased=True)
        (biased_cluster_assignment, biased_cluster_MRFs) = biased_ticc.fit(input_file=fname)

        np.testing.assert_array_equal(np.array(biased_cluster_assignment), np.array(unbiased_cluster_assignment), "Biased assignment is not equel to unbiased assignment!")

    def test_failed_unbiased(self):
        with self.assertRaises(Exception) as context:
            # TICC will fail in Iteration 2, because cluster 9 has only one observation.
            fname = "example_data.txt"
            ticc = TICC(window_size=1, number_of_clusters=50, lambda_parameter=11e-2, beta=600, maxIters=100,
                        threshold=2e-5,
                        write_out_file=False, prefix_string="output_folder/", num_proc=1)
            (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

        self.assertTrue('This is broken {}'.format(context.exception))


if __name__ == '__main__':
    unittest.main()


