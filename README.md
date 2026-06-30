# ML-project-Arthur-Feld

PROJECT MAIN PHASES:

PHASE 1: Understanding tools & problem + problem breakdown
PHASE 2: DEVELOPING MVP - simple linear regression model that learned from the ~60 samples, ideally at the end of week 1
PHASE 3: Improving model into more detailed and realistic model

for each model, split samples into 80% training 20% test
plot prediction graph
compute RMSE = 1Ni=1N(h(xi)- yi)2


PHASE 1:

Going through data sent by professor Redonnet
Understanding : 
how music gen ai works: decomposing frequency into several parameters 
how to properly apply ML in a python code
how to apply with spectrometers as inputs


PHASE 2: 

MVP idea: compute frequency mean between t=0 and t=tmax and define the X’s as this mean. ← does not identify any patterns, purely predicts annoyance score simply from frequency means (simple linear model)

NEXT Model: decomposing spectrometers into d features (where each feature is a mean from t=0 to t=tmax), such that X belongs to R^d. new linear model with this new input: compute parameters such that h(x)=i=1dixi . try to identify which features can be neglected, to clean the model. 

NEXT Model improved: 
