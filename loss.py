from keras.losses import binary_crossentropy
import keras.backend as K

def dice(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    
    TP = K.sum(y_true * y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true) + K.sum(y_pred)
    
    return 2.0 * TP / (union + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1. - dice(y_true, y_pred)

def get_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
