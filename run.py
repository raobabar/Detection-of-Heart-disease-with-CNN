from heart import app
from keras import backend as K
from importlib import reload
import os

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

if __name__ == '__main__':
    set_keras_backend("theano")
    app.run(debug=True, port=5060)