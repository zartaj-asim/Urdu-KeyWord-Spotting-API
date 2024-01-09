from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from sklearn import mixture
import joblib

GmmDimensions = 45
dimensions = 13
windowSize = 0.010
StepSizeBwframes = 0.01

# Load the training data
(rate, sig) = wav.read("CompleteDataGmm.wav")
mfcc_feat = mfcc(sig, rate, windowSize, StepSizeBwframes, dimensions)

# Train the GMM model
gmix = mixture.GaussianMixture(n_components=GmmDimensions, covariance_type='full')
gmix.fit(mfcc_feat)

# Save the trained GMM model to a file
joblib.dump(gmix, "pretrained_gmm_model.joblib")
