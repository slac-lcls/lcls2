import os
import unittest
from mpi4py import MPI
from psana2 import DataSource

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set environment variables for MPI execution
os.environ["PS_SRV_NODES"] = "1"
os.environ["PS_SMD_N_EVENTS"] = "1"

# Global variable to store received data for validation
received_data = []

# Define the callback function
def smd_callback(data_dict):
    """Callback function to process data dictionary."""
    print(f"Rank {rank}: Received data with keys: {data_dict.keys()}")
    received_data.append(data_dict)  # Store received data for later validation

# Define the test class
class TestSmallDataCallbacks(unittest.TestCase):

    def setUp(self):
        """Common setup for both tests: Define the experiment, run, and test data."""
        self.exp = 'xpptut15'
        self.runnum = 14
        self.xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')
        self.mydata = {
            "int_value": 42,
            "nested_dict": {"inner_key": "inner_value"},
            "string_value": "test_string"
        }

    def test_callback_fails_with_filename(self):
        """Test that the callback fails when filename is provided."""
        ds = DataSource(exp=self.exp, run=self.runnum, dir=self.xtc_dir)

        with self.assertRaises(Exception, msg="Callback should fail when filename is given."):
            smd = ds.smalldata(filename="test_file.h5", callbacks=[smd_callback])

            for run in ds.runs():
                for evt in run.events():
                    smd.event(evt, mydata=self.mydata)

            smd.done()

    def test_callback_passes_without_filename(self):
        """Test that the callback passes and receives correct data when filename is not provided."""
        global received_data
        received_data = []  # Reset before test

        ds = DataSource(exp=self.exp, run=self.runnum, dir=self.xtc_dir)
        smd = ds.smalldata(callbacks=[smd_callback])  # No filename given

        for run in ds.runs():
            for evt in run.events():
                smd.event(evt, mydata=self.mydata)

        smd.done()  # Should complete without errors

        # Validate that the received data matches expected mydata
        self.assertGreater(len(received_data), 0, "No data received in callback.")
        # Extract 'mydata' from received data and compare with expected values
        for received in received_data:
            received_filtered = {k: v for k, v in received.get("mydata", {}).items() if k != "timestamp"}
            self.assertEqual(received_filtered, self.mydata, "Received data does not match expected values (ignoring timestamp).")

# Run the tests
if __name__ == "__main__":
    unittest.main()
