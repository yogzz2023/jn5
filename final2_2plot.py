import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        """Initialize filter state."""
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        """Initialize measurement for filtering."""
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        """Predict step of the Kalman filter."""
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self, report):
        """Update step of the Kalman filter using the associated report."""
        Inn = report - np.dot(self.H, self.Sf)  # Calculate innovation using associated report
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = np.arcsin(z / r) * 180 / np.pi
    az = np.arctan2(y, x) * 180 / np.pi
    az[az < 0] += 360  # Adjust azimuth to be in the range [0, 360)
    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    r = []
    az = []
    el = []
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       
        
        az[i] = az[i] * 180 / 3.14 
        

        if az[i] < 0.0:
            az[i] = (360 + az[i])
    
        if az[i] > 360:
            az[i] = (az[i] - 360)   
  
    return r, az, el

def read_measurements_from_csv(file_path):
    """Read measurements from CSV file."""
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

def group_measurements_into_tracks(measurements):
    """Group measurements into tracks."""
    tracks = []
    used_indices = set()
    for i, (x_base, y_base, z_base, mt_base) in enumerate(measurements):
        if i in used_indices:
            continue
        track = [(x_base, y_base, z_base, mt_base)]
        used_indices.add(i)
        for j, (x, y, z, mt) in enumerate(measurements):
            if j in used_indices:
                continue
            if abs(mt - mt_base) < 50:
                track.append((x, y, z, mt))
                used_indices.add(j)
        tracks.append(track)
    return tracks

def is_valid_hypothesis(hypothesis):
    """Check if a hypothesis is valid."""
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    """Calculate Mahalanobis distance."""
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def perform_clustering_hypothesis_association(tracks, reports, cov_inv):
    """Perform clustering, hypothesis generation, and association."""
    clusters = []
    for report in reports:
        distances = [np.linalg.norm(np.array(track[:3]) - report) for track in tracks]
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < chi2_threshold:
            clusters.append([min_distance_idx])
    print("Clusters:", clusters)
    hypotheses = []
    for cluster in clusters:
        num_tracks = len(cluster)
        base = len(reports) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in cluster:
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_idx, report_idx - 1))

            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)


    if not hypotheses:  # If no hypotheses were generated
        return [], []

    probabilities = calculate_probabilities(hypotheses, tracks, reports, cov_inv)
    # print("Probabilities of Hypotheses:", probabilities)
    return hypotheses, probabilities


def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    """Calculate probabilities for each hypothesis."""
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(np.array(tracks[track_idx][:3]), reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance ** 2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def find_max_associations(hypotheses, probabilities, reports):
    """Find the most likely association for each report."""
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs


def plot_track_data(predicted_states, updated_states):
    
    csv_file_predicted = "ttk_84_2.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

    number = 1000
    result = np.divide(A[0], number)
    # Extract the values from predicted_states and updated_states
    times_predicted, ranges_predicted, azimuths_predicted, elevations_predicted = [], [], [], []
    times_updated, ranges_updated, azimuths_updated, elevations_updated = [], [], [], []

    # for state in predicted_states:
    #     times_predicted.append(state[0])
    #     ranges_predicted.append(state[1])
    #     azimuths_predicted.append(state[2])
    #     elevations_predicted.append(state[3])

    for state in updated_states:
        times_updated.append(state[0])
        ranges_updated.append(state[1])
        azimuths_updated.append(state[2])
        elevations_updated.append(state[3])

    # # Check lengths before plotting
    # print(f"Length of times_predicted: {len(times_predicted)}, ranges_predicted: {len(ranges_predicted)}, azimuths_predicted: {len(azimuths_predicted)}, elevations_predicted: {len(elevations_predicted)}")
    # print(f"Length of times_updated: {len(times_updated)}, ranges_updated: {len(ranges_updated)}, azimuths_updated: {len(azimuths_updated)}, elevations_updated: {len(elevations_updated)}")

    plt.figure(figsize=(12, 6))  # Plot Range vs Time
    # if len(times_predicted) == len(ranges_predicted) and len(times_updated) == len(ranges_updated):
    plt.plot(times_predicted, ranges_predicted, label='Predicted Range', color='green', marker='o')
    plt.plot(filtered_values_csv[:, 0], result, label='Filtered Range (Track ID 31)', color='red', marker='*')
    plt.plot(times_updated, ranges_updated, label='Updated Range', color='blue', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Range')
    plt.title('Range vs Time')
    plt.legend()

    plt.figure(figsize=(12, 6))  # Plot Azimuth vs Time
    # if len(times_predicted) == len(azimuths_predicted) and len(times_updated) == len(azimuths_updated):
    plt.plot(times_predicted, azimuths_predicted, label='Predicted Azimuth', color='blue', marker='o')
    plt.plot(filtered_values_csv[:, 0], A[1], label='Filtered Azimuth (Track ID 31)', color='red', marker='*')
    plt.plot(times_updated, azimuths_updated, label='Updated Azimuth', color='blue', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Azimuth')
    plt.title('Azimuth vs Time')
    plt.legend()

    plt.figure(figsize=(12, 6))  # Plot Elevation vs Time
    # if len(times_predicted) == len(elevations_predicted) and len(times_updated) == len(elevations_updated):
    plt.plot(times_predicted, elevations_predicted, label='Predicted Elevation', color='blue', marker='o')
    plt.plot(filtered_values_csv[:, 0], A[2], label='Filtered Elevation (Track ID 31)', color='red', marker='*')
    plt.plot(times_updated, elevations_updated, label='Updated Elevation', color='blue', marker='x')
    plt.xlabel('Time')
    plt.ylabel('Elevation')
    plt.title('Elevation vs Time')
    plt.legend()

    plt.tight_layout()
    plt.show()
def main():
    """Main processing loop."""
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84_2.csv'

    try:
        measurements = read_measurements_from_csv(csv_file_path)
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not measurements:
        print("No measurements found in the CSV file.")
        return

    tracks = group_measurements_into_tracks(measurements)
    cov_inv = np.linalg.inv(np.eye(state_dim))  # Example covariance inverse matrix

    predicted_states = []
    updated_states = []

    for group_idx, track_group in enumerate(tracks):
        print(f"Processing group {group_idx + 1}/{len(tracks)}")

        track_states = []
        reports = []

        for i, (x, y, z, mt) in enumerate(track_group):
            if i == 0:
                # Initialize filter state with the first measurement
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            elif i == 1:
                # Initialize filter state with the second measurement and compute velocity
                prev_x, prev_y, prev_z = track_group[i-1][:3]
                dt = mt - track_group[i-1][3]
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                vz = (z - prev_z) / dt
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
            else:
                kalman_filter.predict_step(mt)
                kalman_filter.initialize_measurement_for_filtering(x, y, z, mt)
                reports.append(np.array([x, y, z]))

            # predicted_states.append((kalman_filter.Meas_Time, np.linalg.norm(kalman_filter.Sf[:3]), *cart2sph(*kalman_filter.Sf[:3])))

        # Perform clustering, hypothesis generation, and association
        hypotheses, probabilities = perform_clustering_hypothesis_association(track_group, reports, cov_inv)

        # # Print hypotheses and their respective probabilities
        # for hypothesis, probability in zip(hypotheses, probabilities):
        #     print(f"Hypothesis: {hypothesis}, Probability: {probability}")

        # Find the most likely association for each report
        max_associations, max_probs = find_max_associations(hypotheses, probabilities, reports)

        for report_idx, track_idx in enumerate(max_associations):
            if track_idx != -1:
                kalman_filter.update_step(reports[report_idx])
                updated_states.append((kalman_filter.Meas_Time, np.linalg.norm(kalman_filter.Sf[:3]), *cart2sph(*kalman_filter.Sf[:3])))

        # Print associated measurements to track
        for i, (report_idx, track_idx) in enumerate(zip(max_associations, track_group)):
            if report_idx != -1:
                print(f"Report {i+1} associated with Track {track_idx}")

        # Print the most likely associated measurements for each target
        for i, (report_idx, prob) in enumerate(zip(max_associations, max_probs)):
            if report_idx != -1:
                print(f"Most likely association for report {i+1}: Track {report_idx} with probability {prob}")

    # Plotting all data together
    plot_track_data(predicted_states, updated_states)

if __name__ == "__main__":
    main()
