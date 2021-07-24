#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from fairseq.models.wav2vec import Wav2VecModel
import soundfile as sf
import numpy as np
from tqdm import tqdm


# In[2]:


audio_input, sample_rate = sf.read('sample.wav')
audio_input_single_channel = audio_input[:,0]
input_batch = []
for i in range(int(len(audio_input_single_channel)/10000)):
    batch = np.array([audio_input_single_channel[i*10000:(i+1)*10000]])
    input_batch.append(torch.from_numpy(batch))


# In[3]:


cp = torch.load('wav2vec_large.pt', map_location=torch.device('cpu'))
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
# model.eval()


# In[ ]:


embeddings = []
for batch in tqdm(input_batch):
    z = model.feature_extractor(batch.float())
    embeddings.append(model.feature_aggregator(z))


# In[ ]:




