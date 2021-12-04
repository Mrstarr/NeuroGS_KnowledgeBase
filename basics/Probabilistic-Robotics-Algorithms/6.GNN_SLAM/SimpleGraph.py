import matplotlib.pyplot as plt
import numpy as np 
import os 

class SimpleGraph: 
    def __init__(self, dataset_name, start_frame, end_frame) -> None:
        self.load_data(dataset_name, start_frame, end_frame)
        self.plot_data()


    def load_data(self, dataset_name, start_frame, end_frame):
        # Loading dataset
        # Barcodes: [Subject#, Barcode#]
        self.barcodes_data = np.loadtxt(dataset_name + "/Barcodes.dat")
        # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
        self.groundtruth_data = np.loadtxt(dataset_name + "/Groundtruth.dat")
        # Landmark ground truth: [Subject#, x[m], y[m]]
        self.landmark_groundtruth_data = np.loadtxt(dataset_name + "/Landmark_Groundtruth.dat")
        # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
        self.measurement_data = np.loadtxt(dataset_name + "/Measurement.dat")
        # Odometry: [Time[s], Subject#, forward_V[m/s], angular _v[rad/s]]
        self.odometry_data = np.loadtxt(dataset_name + "/Odometry.dat")

        # Collect all input data and sort by timestamp
        # Add subject "odom" = -1 for odometry data
        odom_data = np.insert(self.odometry_data, 1, -1, axis=1)
        self.data = np.concatenate((odom_data, self.measurement_data), axis=0)
        self.data = self.data[np.argsort(self.data[:, 0])]

        # Select data according to start_frame and end_frame
        # First frame must be control input
        while self.data[start_frame][1] != -1:
            start_frame += 1
        # Remove all data before start_frame and after the end_timestamp
        self.data = self.data[start_frame:end_frame]
        start_timestamp = self.data[0][0]
        end_timestamp = self.data[-1][0]
        # Remove all groundtruth outside the range
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= end_timestamp):
                break
        self.groundtruth_data = self.groundtruth_data[:i]
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= start_timestamp):
                break
        self.groundtruth_data = self.groundtruth_data[i:]

        # Combine barcode Subject# with landmark Subject#
        # Lookup table to map barcode Subjec# to landmark coordinates
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        # Ground truth data is not used in EKF SLAM
        self.landmark_locations = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_locations[self.barcodes_data[i][1]] = self.landmark_groundtruth_data[i - 5][1:]

        # Lookup table to map barcode Subjec# to landmark Subject#
        # Barcode 6 is the first landmark (1 - 15 for 6 - 20)
        self.landmark_indexes = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_indexes[self.barcodes_data[i][1]] = i - 5

        for key, value in self.landmark_indexes.items(): 
            print("key: " + str(key))
            print("item: " + str(value))

        # Table to record if each landmark has been seen or not
        # Element [0] is not used. [1] - [15] represent for landmark# 6 - 20
        self.landmark_observed = np.full(len(self.landmark_indexes) + 1, False)
        print("observed: " + str(self.landmark_observed))


    def plot_data(self):
        # Clear all
        plt.cla()

        # Ground truth data
        plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2], 'b', label="Robot State Ground truth")

        # States
        #plt.plot(self.states[:, 0], self.states[:, 1], 'r', label="Robot State Estimate")

        # Start and end points
        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2], 'g8', markersize=12, label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2], 'y8', markersize=12, label="End point")

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 6
            plt.text(landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10)
        plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*', label='Landmark Ground Truth')

        """
        # Landmark estimated locations
        estimate_xs = []
        estimate_ys = []
        for i in range(1, len(self.landmark_indexes) + 1):
            if self.landmark_observed[i]:
                estimate_xs.append(self.states[-1][2 * i + 1])
                estimate_ys.append(self.states[-1][2 * i + 2])
                plt.text(estimate_xs[-1], estimate_ys[-1], str(i+5), fontsize=10)
        plt.scatter(estimate_xs, estimate_ys, s=50, c='k', marker='.', label='Landmark Estimate')
        """

        plt.title('GNN SLAM with known correspondences')
        plt.legend()
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))
        plt.pause(1e-16)


if __name__ == "__main__":
    # Dataset 1
    dataset_name = os.path.join("0.Dataset1")
    start_frame = 800
    end_frame = 3200
    # State covariance matrix
    R = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2
    # Measurement covariance matrix
    Q = np.diagflat(np.array([110.0, 110.0, 1e16])) ** 2

    GNN_slam = SimpleGraph(dataset_name, start_frame, end_frame)
    plt.show()