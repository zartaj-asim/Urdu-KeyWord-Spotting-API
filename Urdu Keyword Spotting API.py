from flask import Flask, request, jsonify, render_template, send_file
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import joblib
import math 
import warnings
from scipy.io import wavfile

app = Flask(__name__)

GmmDimensions = 45
dimensions = 13
windowSize = 0.010
StepSizeBwframes = 0.01

# Function to load the pretrained GMM model from a file
def load_gmm_model(file_path):
    return joblib.load(file_path)

# File path for loading the pretrained GMM model
gmm_model_file = "pretrained_gmm_model.joblib"
gmm_model = load_gmm_model(gmm_model_file)

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# Process audio file and return Gaussian probability vectors
def process_audio_file(file_path):
    (rate, sig) = wav.read(file_path)
    mfcc_feat = mfcc(sig, rate, windowSize, StepSizeBwframes, dimensions)
    samples = mfcc_feat
    GausianProbablitiesVectors = gmm_model.predict_proba(samples)
    return GausianProbablitiesVectors

@app.route('/temp/<filename>')
def serve_audio(filename):
    temp_dir = "temp"
    audio_path = os.path.join(temp_dir, filename)
    return send_file(audio_path, mimetype="audio/wav")


@app.route('/', methods=['GET', 'POST'])
def index():
    haystack_url = None  # Initialize the haystack_url variable
    needle_url = None
    startIndex = []     # Initialize startIndex list
    endIndex = []       # Initialize endIndex list
    NumberOFWindows = 0  # Assign a default value to NumberOFWindows

    if request.method == 'POST':
        if 'needle' not in request.files or 'haystack' not in request.files:
            return render_template('index.html', error='Missing audio files (needle or haystack)')

        # Check and create 'temp' directory if it doesn't exist
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the audio files temporarily
        needle_path = os.path.join(temp_dir, "needle.wav")
        haystack_path = os.path.join(temp_dir, "haystack.wav")
        needle_file = request.files['needle']
        haystack_file = request.files['haystack']
        needle_file.save(needle_path)
        haystack_file.save(haystack_path)

        # Process the audio files and calculate Gaussian probability vectors
        NeedleVector = process_audio_file(needle_path)
        HeystackVector = process_audio_file(haystack_path)

        # Perform DTW calculations and sorting of results
        distances, startIndex, endIndex, NumberOFWindows = perform_dtw(NeedleVector, HeystackVector)

        # Sort and get only the top 10 distances and their indices
        bubbleSort(distances, startIndex, endIndex)

        # Verify file paths and remove the temporary files
        print("Needle Path:", needle_path)
        print("Haystack Path:", haystack_path)
        

        # Generate a URL for the haystack audio file to pass to the HTML page
        haystack_url = 'http://127.0.0.1:5000/temp/haystack.wav'
        needle_url = 'http://127.0.0.1:5000/temp/needle.wav'

    # Prepare the data to pass to the index.html page
    result_data = {
        'total_windows': NumberOFWindows,
        'needle_occurrences': list(zip(startIndex, endIndex)),
        'haystack_url': haystack_url, 
        'needle_url': needle_url,
    }

    # Return the index.html template with the data
    return render_template('index.html', **result_data)



def CalculateDistanceMatrixViaDotProduct(HeystackVector, NeedleVector, distanceMatrix):
    for i in range(len(HeystackVector)):
        for j in range(len(NeedleVector)):
            H = HeystackVector[i]
            N = NeedleVector[j]
            cccc = np.dot(H, N)
            if cccc > 0:  # Add this check to prevent math domain error
                cccc = -(math.log10(cccc))
            else:
                cccc = float('inf')  # Set to a large value if cccc is not positive
            distanceMatrix[i, j] = cccc

def CalculateAccumlatedCost(HeystackVector, NeedleVector, distanceMatrix, accumulated_cost):
    for i in range(1, len(HeystackVector)):
        for j in range(1, len(NeedleVector)):
            accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distanceMatrix[i, j]

def path_cost(x, y, accumulated_cost, distances):
    path = [[len(x) - 1, len(y) - 1]]
    cost = 0
    i = len(x) - 1
    j = len(y) - 1
    while i > 0 and j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append([j, i])
    path.append([0, 0])
    for [y, x] in path:
        cost = cost + distances[x, y]
    return path, cost

# Rest of the code remains the same as provided in the previous response

def perform_dtw(needle_vectors, haystack_vectors):
    distances = []
    startIndex = []
    endIndex = []
    stepUp = 15
    step = 5
    MainCounter = 0
    NumberOFWindows = 0
    while MainCounter < haystack_vectors.shape[0]:
        temp = needle_vectors.shape[0] + MainCounter
        if temp <= haystack_vectors.shape[0]:
            windowSize = temp - MainCounter
            HeystackTempVector = np.zeros(shape=(windowSize, GmmDimensions))
            counter = 0
            i = MainCounter
            startIndex.append(i)
            while i < temp:
                res = haystack_vectors[i, :]
                res = res.reshape(1, -1)
                HeystackTempVector[counter, :] = res
                i += 1
                counter += 1

            MainCounter = MainCounter + step
            endIndex.append(i)
            NumberOFWindows = NumberOFWindows + 1

            NeedleVector = np.array(needle_vectors)
            HeystackVector = np.array(HeystackTempVector)

            distanceMatrix = np.zeros((len(HeystackVector), len(NeedleVector)))
            CalculateDistanceMatrixViaDotProduct(HeystackVector, NeedleVector, distanceMatrix)

            accumulated_cost = np.zeros((len(HeystackVector), len(NeedleVector)))
            CalculateAccumlatedCost(HeystackVector, NeedleVector, distanceMatrix, accumulated_cost)

            x = HeystackVector
            y = NeedleVector
            distancesM = distanceMatrix
            path, cost = path_cost(x, y, accumulated_cost, distancesM)
            distances.append(cost)

        else:
            print("discarded frames ", temp - haystack_vectors.shape[0])
            MainCounter = haystack_vectors.shape[0]

    # Now that we have the distances, startIndex, and endIndex, let's print them here
    ListSize = len(distances)
    for i in range(ListSize):
        print(f"Distance {i}: {distances[i]}")
        print(f"Start Index {i}: {startIndex[i]}")
        print(f"End Index {i}: {endIndex[i]}")

    return distances, startIndex, endIndex, NumberOFWindows


def bubbleSort(distances, startIndex, endIndex):
    for passnum in range(len(distances) - 1, 0, -1):
        for i in range(passnum):
            if distances[i] > distances[i+1]:
                temp = distances[i]
                distances[i] = distances[i+1]
                distances[i+1] = temp

                tempStart = startIndex[i]
                startIndex[i] = startIndex[i+1]
                startIndex[i+1] = tempStart

                tempEnd = endIndex[i]
                endIndex[i] = endIndex[i+1]
                endIndex[i+1] = tempEnd



if __name__ == '__main__':
    app.run(port=5000, debug=True)
