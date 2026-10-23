from tensorflow import keras


def SimpleMLP(num_of_features: int = 3, num_of_classes: int = 6):
    model = keras.Sequential([
        keras.layers.Dense(32, input_shape=(num_of_features,), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(num_of_classes, activation='softmax')])
    return model
