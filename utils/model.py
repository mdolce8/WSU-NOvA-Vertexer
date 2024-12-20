# model.py

import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.python.client import device_lib
import time
from typing import Tuple

# Useful bits for the model
class Hardware:
    @staticmethod
    def check_gpu_status():
        print('========================================')
        # GPU Test
        # see if GPU is recognized
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print('is built with cuda?: ', tf.test.is_built_with_cuda())
        print('is gpu available?: ', tf.config.list_physical_devices('GPU'))
        print('session: ', sess)
        print('list_local_devices(): ', device_lib.list_local_devices())
        print('========================================')

        #     +-----------------------------------------------------------------------------+
        #     | NVIDIA-SMI 440.64.00    Driver Version: 440.64.00    CUDA Version: 10.2     |
        #     |-------------------------------+----------------------+----------------------+
        #     | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        #     | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        #     |===============================+======================+======================|
        #     |   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
        #     | N/A   33C    P0    34W / 250W |  15431MiB / 16160MiB |      0%      Default |
        #     +-------------------------------+----------------------+----------------------+
        #     |   1  Tesla V100-PCIE...  Off  | 00000000:D8:00.0 Off |                    0 |
        #     | N/A   34C    P0    34W / 250W |      0MiB / 16160MiB |      5%      Default |
        #     +-------------------------------+----------------------+----------------------+
        #
        #     +-----------------------------------------------------------------------------+
        #     | Processes:                                                       GPU Memory |
        #     |  GPU       PID   Type   Process name                             Usage      |
        #     |=============================================================================|
        #     |    0    152871      C   python3                                    15419MiB |
        #     +-----------------------------------------------------------------------------+
        return sess

class Config:
    @staticmethod
    def create_test_train_val_datasets(map_xz, map_yz, vtx_coords, test_size=0.25, val_size=0.1, random_state=101) -> Tuple[dict, dict, dict]:
        """
        :param map_xz: cvnmap for XZ view (features)
        :param map_yz: cvnmap for YZ view (features)
        :param vtx_coords: (x,y,z) true coordinates (labels)
        :param test_size: fraction of test size data
        :param val_size: fraction of validation size data
        :param random_state: def. 101
        :return: train, val, test dictionaries
        """
        map_tr_xz, map_te_xz, map_tr_yz, map_te_yz, vtx_tr, vtx_te = train_test_split(map_xz,
                                                                                      map_yz,
                                                                                      vtx_coords,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state)
        test = {'xz': map_te_xz, 'yz': map_te_yz, 'vtx': vtx_te}

        # Divide training data again, for a validation set.
        map_trf_xz, map_val_xz, map_trf_yz, map_val_yz, vtx_trf, vtx_val = train_test_split(map_tr_xz,
                                                                                            map_tr_yz,
                                                                                            vtx_tr,
                                                                                            test_size=val_size,
                                                                                            random_state=random_state)

        train = {'xz': map_trf_xz, 'yz': map_trf_yz, 'vtx': vtx_trf}
        val = {'xz': map_val_xz, 'yz': map_val_yz, 'vtx': vtx_val}
        return train, val, test

# or assemble_model_conv_inputs() ?
    @staticmethod
    def create_conv2d_branch_model_single_view() -> Sequential:
        """
        Create the branch model for the XZ or YZ view.
        :return: Sequential() model
        """
        m = Sequential([
            Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=(100, 80, 1)),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
        ])
        return m

    @staticmethod
    def assemble_model_output(model_xz, model_yz) -> Model:
        input_shape = (100, 80, 1)
        input_xz = Input(shape=input_shape)
        input_yz = Input(shape=input_shape)

        # get outputs from the Sequential models, one from each view.
        output_xz = model_xz(input_xz)
        output_yz = model_yz(input_yz)

        # concatenate outputs
        concatenated = Concatenate(axis=-1)([output_xz, output_yz])

        # add additional dense layers and output
        dense_layer_1 = Dense(256, activation='relu')(concatenated)
        d1 = Dropout(0.3)(dense_layer_1)
        dense_layer_2 = Dense(256, activation='relu')(dense_layer_1)
        d2 = Dropout(0.3)(dense_layer_2)
        dense_layer_3 = Dense(256, activation='relu')(dense_layer_2)
        output = Dense(3, activation='linear')(dense_layer_3)

        return Model(inputs=[input_xz, input_yz], outputs=output)

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    @staticmethod
    def compile_model(model_output, loss="logcosh", metrics=None):
        """
        Compile the model output.
        :param model_output: the model you would like to compile
        :param loss: the loss to use
        :param metrics: the metrics to use. Default is ["mse", "mae"]
        :return: None
        """
        # Logcosh calculated independently for x, y, z. Then summed
        if metrics is None:
            metrics = ["mse", "mae"]
        model_output.compile(loss=loss,
                             optimizer='adam',
                             metrics=metrics)  # loss was 'mse' then 'mae'
        print('Selected Loss Function: ', loss)
        print('Selected Metrics: ', metrics)
        return None

def train_model(model_output,
                data_training,
                data_validation,
                epochs,
                batch_size=32):
    """
    Perform fit() func, i.e. do the training.
    :param model_output: Model (expected output)
    :param data_training: dict (training data, should already be divided for validation)
    :param data_validation: dict (validation data, should already be divided)
    :param epochs: int (from args.epoch)
    :param batch_size: int (how many maps should model see)
    :return: History.history
    """
    start = time.time()
    history = model_output.fit(
        x={'data_train_xz': data_training['xz'], 'data_train_yz': data_training['yz']},
        y=data_training['vtx'],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=({'data_val_xz': data_validation['xz'], 'data_val_yz': data_validation['yz']}, data_validation['vtx']))

    stop = time.time()
    elapsed = (stop - start) / 60
    print(f'Time to train: {elapsed:.2f} minutes.')
    return history

def evaluate_model(model_output, data_testing, filtered_events, evaluate_dir):
    """
    Evaluate the model on testing data.
    :param model_output: Model (output from model)
    :param data_testing: dict (testing data, should already be divided for testing)
    :param filtered_events: dict {"keep": array, "drop: array"} in that order
    :param evaluate_dir: str (directory to save evaluation results)
    :return: evaluation: array (single values for each metric)
    """
    start_eval = time.time()
    print('Evaluation on the test set...')
    evaluation = model_output.evaluate(
        x={'data_test_xz': data_testing['xz'], 'data_test_yz': data_testing['xz']},
        y=data_testing['vtx'])
    stop_eval = time.time()
    print('Test Set Evaluation: {}'.format(evaluation))
    print('Evaluation time: ', stop_eval - start_eval)

    # NOTE: evaluation only returns ONE number for each metric , and one for the loss, so just write to txt file.
    if not os.path.exists(evaluate_dir):
        os.makedirs(evaluate_dir)
        print('created dir: {}'.format(evaluate_dir))
    else:
        print('dir already exists: {}'.format(evaluate_dir))
    with open(f'{evaluate_dir}/evaluation_results.txt', 'w') as f:
        f.write(f"Test Loss: {evaluation[0]}\n")
        f.write(f"Test MSE: {evaluation[1]}\n")
        f.write('Passed: {}\n'.format(len(filtered_events['keep'])))
        f.write('Dropped: {}\n'.format(len(filtered_events['drop'])))
    print('Saved evaluation to: ', evaluate_dir + '/evaluation_results.txt')
    return evaluation
