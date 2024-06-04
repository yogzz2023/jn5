import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
import mplcursors


class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))
        self.Pf = np.eye(6)
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20
        self.H = np.eye(3, 6)
        self.R = np.eye(3)
        self.Meas_Time = 0
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self, report):
        Inn = report - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = np.arcsin(z / r) * 180 / np.pi
    az = np.arctan2(y, x) * 180 / np.pi
    az[az < 0] += 360
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
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            mr = float(row[7])
            ma = float(row[8])
            me = float(row[9])
            mt = float(row[10])
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def group_measurements_into_tracks(measurements):
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
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def perform_clustering_hypothesis_association(tracks, reports, cov_inv):
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

    if not hypotheses:
        return [], []

    probabilities = calculate_probabilities(hypotheses, tracks, reports, cov_inv)
    return hypotheses, probabilities

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
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
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

def plot_track_data(updated_states):
    csv_file_predicted = "ttk_84_2.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

    number = 1000
    result = np.divide(A[0], number)

    times, ranges, azimuths, elevations = zip(*updated_states)

    # print(f"Length of times: {len(times)}")
    # print(f"Length of ranges: {len(ranges)}")
    # print(f"Length of azimuths: {len(azimuths)}")
    # print(f"Length of elevations: {len(elevations)}")

    plt.figure(figsize=(12, 6))
    # plt.scatter(times, ranges, label='Range', color='blue', marker='o')
    plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
    plt.xlabel('Time', color='black')
    plt.ylabel('Range (r)', color='black')
    plt.title('Range vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)

    plt.figure(figsize=(12,6))
    # plt.scatter(times, azimuths, label='Azimuth', color='blue', marker='o')
    plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)


    plt.figure(figsize=(12, 6))
    # plt.scatter(times, elevations, label='Elevation', color='blue', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
    plt.xlabel('Time', color='black')
    plt.ylabel('Elevation (el)', color='black')
    plt.title('Elevation vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
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
    cov_inv = np.linalg.inv(np.eye(state_dim))

    predicted_states = []
    updated_states = []

    for group_idx, track_group in enumerate(tracks):
        print(f"Processing group {group_idx + 1}/{len(tracks)}")

        track_states = []
        reports = []

        for i, (x, y, z, mt) in enumerate(track_group):
            if i == 0:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            elif i == 1:
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

            range_, azimuth, elevation = cart2sph(*kalman_filter.Sf[:3])
            predicted_states.append((kalman_filter.Meas_Time, range_, azimuth, elevation))

        hypotheses, probabilities = perform_clustering_hypothesis_association(track_group, reports, cov_inv)

        max_associations, max_probs = find_max_associations(hypotheses, probabilities, reports)

        for report_idx, track_idx in enumerate(max_associations):
            if track_idx != -1:
                kalman_filter.update_step(reports[report_idx])
                range_, azimuth, elevation = cart2sph(*kalman_filter.Sf[:3])
                updated_states.append((kalman_filter.Meas_Time, range_, azimuth, elevation))

    # for state in updated_states:
    #     print(state)

    plot_track_data(updated_states)

if __name__ == "__main__":
    main()
