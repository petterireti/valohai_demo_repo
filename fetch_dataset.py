import numpy as np
import valohai

valohai.prepare(
    step='fetch-data',
    image='python:3.9',
    default_inputs={
        'dataset': 'https://valohaidemo.blob.core.windows.net/mnist/mnist.npz',
    },
)

print('Loading data')
with np.load(valohai.inputs('dataset').path(), allow_pickle=True) as file:
    x_train, y_train = file['x_train'], file['y_train']
    x_test, y_test = file['x_test'], file['y_test']

print('Saving fecthed data')
path = valohai.outputs().path('fetched_dataset.npz')
np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)