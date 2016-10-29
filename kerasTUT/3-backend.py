"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 3 - backend


"""
Details are showing in the video.

----------------------
Method 1:
If you have run Keras at least once, you will find the Keras configuration file at:

~/.keras/keras.json

If it isn't there, you can create it.

The default configuration file looks like this:

{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}

Simply change the field backend to either "theano" or  "tensorflow",
and Keras will use the new configuration next time you run any Keras code.
----------------------------
Method 2:

define this before import keras:

>>> import os
>>> os.environ['KERAS_BACKEND']='theano'
>>> import keras
Using Theano backend.

"""

