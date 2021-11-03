#!/usr/bin/env python
# coding: utf-8

# In[1]:


#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


# In[2]:


#hide
from fastbook import *
from fastai.vision.widgets import *


# Let's check that the file exists, by using the `ls` method that fastai adds to Python's `Path` class:

# Stroke Classifier!
# 
# Check whether or not your CT Brain imaging is that of a normal brain scan, a haemorrhagic stroke or an ischaemic stroke. You're in the right place.
# Take a picture of the CT brain image and click 'upload' to classify it. (Note - this won't give details on any other pathology eg tumours present in the image).
# It will only give a sensible answer for normal CT brain scans, ischaemic strokes or haemorrhagic strokes                                   

# In[3]:


path = Path()
#path.ls(file_exts='.pkl')


# In[4]:


learn_inf = load_learner(path/'export.pkl', cpu=True)


# In[5]:


btn_upload = widgets.FileUpload()


# In[6]:


#hide_output
#there is a kind of widget called "Output". the "widgets.Output" is basically something that you can fill in later
#the output placeholder i.e. "out.pl" is what determines where your output will be displayed
out_pl = widgets.Output()


# In[7]:


lbl_pred = widgets.Label()


# In[8]:


def on_click(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = (f'Prediction: {pred} ({probs[pred_idx]:.04f}')


# You can test the button now by pressing it, and you should see the image and predictions update automatically!
# 
# We can now put them all in a vertical box (`VBox`) to complete our GUI:

# In[9]:


#hide
#Putting back btn_upload to a widget for next cell
btn_upload.observe(on_click, names=['data'])


# In[10]:


#hide_output
#Now, we've run the VBox, it contians all the pieces we inputted
#so, this is our app. Select an 
display(VBox([widgets.Label('Select your image!'), 
      btn_upload, out_pl, lbl_pred]))


# ### Turning Your Notebook into a Real App

# In[44]:


#hide
get_ipython().system('pip install voila')
get_ipython().system('jupyter serverextension enable --sys-prefix voila ')


# In[15]:


path = Path()
path.ls(file_exts='.pkl')


# In[ ]:





# In[ ]:




