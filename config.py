import os 
from os.path import join,dirname, realpath
from keras.models import load_model

basedir = os.path.abspath(os.path.dirname(__file__))

CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

#SQLALCHEMY CONFIGURATIONS
if os.environ.get('DATABASE_URL') is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'heart.db')
else:
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

#FLASK-BASICAUTH CONFIGURATIONS
BASIC_AUTH_USERNAME = 'admin'
BASIC_AUTH_PASSWORD = 'admin'

#Uploads 
UPLOAD_FOLDER_LOGO = join(dirname(realpath(__file__)), 'static/images/')

#ANN MODEL 
classifier = load_model('my_model.h5')