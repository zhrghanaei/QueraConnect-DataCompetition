import pandas as pd
df = pd.read_csv('drive/MyDrive/football data/Captain Tsubasa/train.csv')
test_df = pd.read_csv('drive/MyDrive/football data/Captain Tsubasa/test.csv') 

### clean data and set labels
def set_label(x):
  if x == 'گُل' or x == 'گُل به خودی':
    return 1
  else:
    return 0

train_df = df[list(test_df)].copy()
train_df.dropna(inplace = True)
train_df['target'] = df.apply(lambda row : set_label(row['outcome']), axis = 1)

### train model
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode='binary')

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

BATCH_SIZE = 32

val_dataframe = train_df.sample(frac=0.2, random_state=1337)
train_dataframe = train_df.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# Categorical feature encoded as string
playType = keras.Input(shape=(1,), name="playType", dtype="string")
bodyPart = keras.Input(shape=(1,), name="bodyPart", dtype="string")
interferenceOnShooter = keras.Input(shape=(1,), name="interferenceOnShooter", dtype="string")

# Numerical features
# minute = keras.Input(shape=(1,), name="minute")
# second = keras.Input(shape=(1,), name="second")
x = keras.Input(shape=(1,), name="x")
y	 = keras.Input(shape=(1,), name="y")
interveningOpponents = keras.Input(shape=(1,), name="interveningOpponents")
interveningTeammates = keras.Input(shape=(1,), name="interveningTeammates")

all_inputs = [
    # minute,
    # second,
    x,
    y,
    playType,
    bodyPart,
    interveningOpponents,
    interveningTeammates,
    interferenceOnShooter]

# String categorical features
playType_encoded = encode_categorical_feature(playType, "playType", train_ds, True)
bodyPart_encoded = encode_categorical_feature(bodyPart, "bodyPart", train_ds, True)
interferenceOnShooter_encoded = encode_categorical_feature(interferenceOnShooter, "interferenceOnShooter", train_ds, True)

# Numerical features
# minute_encoded = encode_numerical_feature(minute, "minute", train_ds)
# second_encoded = encode_numerical_feature(second, "second", train_ds)
x_encoded = encode_numerical_feature(x, "x", train_ds)
y_encoded = encode_numerical_feature(y, "y", train_ds)
interveningOpponents_encoded = encode_numerical_feature(interveningOpponents, "interveningOpponents", train_ds)
interveningTeammates_encoded = encode_numerical_feature(interveningTeammates, "interveningTeammates", train_ds)

all_features = layers.concatenate(
    [
        # minute_encoded,
        # second_encoded,
        x_encoded,
        y_encoded,
        playType_encoded,
        bodyPart_encoded,
        interveningOpponents_encoded,
        interveningTeammates_encoded,
        interferenceOnShooter_encoded,
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
# x = layers.Dense(250, activation="relu")(all_features)
# x = layers.Dense(100, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=50, validation_data=val_ds)
# print(model.summary())

### make predictions
test_df["interferenceOnShooter"].fillna('متوسط', inplace = True)
predictions = model.predict(dict(test_df))
out_df = pd.DataFrame(predictions)
out_df.columns = ['prediction']
out_df.to_csv('output.csv')