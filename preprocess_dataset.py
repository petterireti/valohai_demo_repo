import numpy as np
import valohai

valohai.prepare(
    step='preprocess-dataset',
    image='python:3.9',
    default_inputs={
        'fetched_dataset': '',
    },
)

input_path = valohai.inputs('fetched_dataset').path()
with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print('Preprocessing (resizing) data')
x_train, x_test = x_train / 255.0, x_test / 255.0

print('Saving preprocessed data')
path = valohai.outputs().path('preprocessed_mnist.npz')
np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
