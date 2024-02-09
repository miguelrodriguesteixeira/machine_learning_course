import pandas as pd, numpy as np
from scipy.io.arff import loadarff

# Load and prepare data
data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
X,y = df.drop('class', axis=1), df['class']


