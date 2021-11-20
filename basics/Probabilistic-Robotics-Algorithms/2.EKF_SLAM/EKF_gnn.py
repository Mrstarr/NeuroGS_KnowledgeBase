import numpy as np
import os

def load_data(dataset, start_frame, end_frame):
    # Loading dataset
    # Barcodes: [Subject#, Barcode#]
    barcodes_data = np.loadtxt(dataset + "/Barcodes.dat")
    # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
    groundtruth_data = np.loadtxt(dataset + "/Groundtruth.dat")
    # Landmark ground truth: [Subject#, x[m], y[m]]
    landmark_groundtruth_data = np.loadtxt(dataset + "/Landmark_Groundtruth.dat")
    # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
    measurement_data = np.loadtxt(dataset + "/Measurement.dat")
    # Odometry: [Time[s], Subject#, forward_V[m/s], angular _v[rad/s]]
    odometry_data = np.loadtxt(dataset + "/Odometry.dat")

    # Collect all input data and sort by timestamp
    # Add subject "odom" = -1 for odometry data
    odom_data = np.insert(odometry_data, 1, -1, axis=1)
    data = np.concatenate((odom_data, measurement_data), axis=0)
    data = data[np.argsort(data[:, 0])]
    # ---- now data is sorted by time, ascending ----

    # Select data according to start_frame and end_frame
    # Fisrt frame must be control input
    while data[start_frame][1] != -1:
        start_frame += 1
    # --- start_frame is 801 now ----

    # Remove all data before start_frame and after the end_timestamp
    data = data[start_frame:end_frame]
    # --- data: 801 - 3200

    start_timestamp = data[0][0]
    end_timestamp = data[-1][0]
    
    # Remove all groundtruth outside the range
    for i in range(len(groundtruth_data)):
        if (groundtruth_data[i][0] >= end_timestamp):
            break
    groundtruth_data = groundtruth_data[:i]
    for i in range(len(groundtruth_data)):
        if (groundtruth_data[i][0] >= start_timestamp):
            break
    groundtruth_data = groundtruth_data[i:]
    # --- now groundtruth_data is (18285, 4)
    # ---- 2021 Nov 20 ----

    # Combine barcode Subject# with landmark Subject#
    # Lookup table to map barcode Subjec# to landmark coordinates
    # [x[m], y[m], x std-dev[m], y std-dev[m]]
    # Ground truth data is not used in EKF SLAM
    landmark_locations = {}
    for i in range(5, len(barcodes_data), 1):
        landmark_locations[barcodes_data[i][1]] = landmark_groundtruth_data[i - 5][1:]

    # Lookup table to map barcode Subjec# to landmark Subject#
    # Barcode 6 is the first landmark (1 - 15 for 6 - 20)
    landmark_indexes = {}
    for i in range(5, len(barcodes_data), 1):
        landmark_indexes[barcodes_data[i][1]] = i - 4

    # Table to record if each landmark has been seen or not
    # Element [0] is not used. [1] - [15] represent for landmark# 6 - 20
    landmark_observed = np.full(len(landmark_indexes) + 1, False)



if __name__ == "__main__":
    # Dataset 1
    dataset = os.path.join("0.Dataset1")
    start_frame = 800
    end_frame = 3200
    # State covariance matrix
    R = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2
    # Measurement covariance matrix
    Q = np.diagflat(np.array([110.0, 110.0, 1e16])) ** 2

    ekf_slam = load_data(dataset, start_frame, end_frame)

