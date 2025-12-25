# Noise detection with AI
The Noise Pollution Detector is an AI-powered system designed to identify, measure, and map noise levels in urban environments. It analyzes audio recordings collected from sensors or smartphones and uses machine-learning models to classify different types of noise — such as traffic, construction, or human activity — and to detect high-noise hotspots. The system helps city planners, researchers, and communities better understand noise pollution patterns, supporting data-driven decisions to improve urban living conditions and public well-being

<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Final project for the Building AI course

## Summary

The Noise Pollution Detector uses AI to identify and map high-noise areas in urban environments. By analyzing sound data from different locations, the system helps city planners and communities reduce noise pollution and improve quality of life. Building AI course project.


## Background

Noise pollution is a common problem in cities and can affect health, sleep quality, and well-being. Urban areas often have traffic, construction, and other sources of noise that are difficult to monitor manually. This project aims to:

Detect high-noise areas automatically

Identify patterns and trends in urban noise levels

Support city planning and community interventions

My motivation comes from observing noisy neighborhoods and wanting to apply AI to improve living conditions.


## How is it used?
The system collects audio data from sensors or smartphones in different parts of a city. AI models classify the intensity and type of noise (traffic, construction, human activity). City planners or local authorities can use this data to:

Identify noise hotspots

Plan traffic routes or construction schedules

Take targeted actions to reduce noise

Residents can also access summaries of noise levels in their neighborhoods to stay informed.
Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

This is how you create code examples:
```
import librosa
import numpy as np

# Load audio file
y, sr = librosa.load('example_audio.wav', duration=5.0)
# Compute Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# Convert to log scale
log_S = librosa.power_to_db(S, ref=np.max)
```

## Data sources and AI methods
The project uses:

Data sources: Open urban sound datasets (e.g., UrbanSound8K
) or user-contributed recordings

AI techniques:

Audio classification using neural networks (CNNs on spectrograms)

Regression models to predict noise levels over time

Clustering for hotspot detection

Example code snippet for audio preprocessing:

## Challenges

The model cannot reduce noise by itself; it only provides insights

Audio data collection may raise privacy concerns

Predictions may be biased if sensors are not evenly distributed or if certain noise types dominate the dataset
## What next?

Future improvements could include:

Real-time noise monitoring with IoT devices

Community dashboards showing noise levels over time

Integrating with city planning software to suggest actionable interventions

Collaboration with local authorities for targeted noise reduction strategies

## Acknowledgments

UrbanSound8K dataset – CC BY 4.0

Inspiration: projects analyzing environmental noise with AI and open-source audio processing tools

Librosa library for audio analysis in Python
