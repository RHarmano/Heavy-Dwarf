from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model

import os
try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None


pydot.Dot.create(pydot.Dot())


# generate classification model
class_model = Sequential()
class_model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=(25,25,25,1), padding='same'))
class_model.add(Conv3D(64, (3,3,3), activation='relu', padding='same'))
class_model.add(Flatten())
class_model.add(Dense(128, activation='relu'))
class_model.add(Dropout(0.5))
class_model.add(Dense(2))

# generate regression model
regress_model = Sequential()
regress_model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=(25,25,25,1), padding='same'))
regress_model.add(BatchNormalization())
regress_model.add(MaxPool3D((2,2,2,)))
regress_model.add(Conv3D(64, (3,3,3), activation='relu', padding='same'))
regress_model.add(BatchNormalization())
regress_model.add(MaxPool3D((2,2,2,)))
regress_model.add(Flatten())
regress_model.add(Dense(32, activation='relu'))
regress_model.add(BatchNormalization())
regress_model.add(Dropout(0.5))
regress_model.add(Dense(2))


plot_model(class_model, to_file='./Final_Figs/Class_Model.png', show_shapes=True, show_layer_names=True, dpi=1200)