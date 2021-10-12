from keras.models import Model
from keras.layers.core import Dense
from keras.models import Sequential

def getSequentialModel( in_shape, layers ):

    # iteratively create sequential model
    model = Sequential()    
    for idx, layer in enumerate( layers ):

        # connect to inputs
        if idx == 0:
            model.add ( Dense( layer[ 'units' ], input_dim=in_shape, activation=layer[ 'activation' ] ) )
        else:
            model.add ( Dense( layer[ 'units' ], activation=layer[ 'activation' ] ) )

    return model
