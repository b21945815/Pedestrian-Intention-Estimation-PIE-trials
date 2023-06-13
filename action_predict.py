import time
import yaml
from dataUtilities import *
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRU, GRUCell
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class ActionPredict(object):
    """
        A base interface class for prediction models
    """
    def __init__(self, **kwargs):

        # Network parameters
        self._regularizationProcess = regularizers.l2(kwargs['regularization_value'])
        self._num_hidden_units = kwargs['num_hidden_units']

    # Processing images and generate features
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='bbox',
                                     crop_resize_ratio=2):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'bbox' (crop using bounding box coordinates),
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={}\
              \nsave_path={}, ".format(data_type, crop_type, save_path))

        preprocess_input = vgg16.preprocess_input

        convolutional_net = vgg16.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                set_id = imp.split('\\')[-3]
                vid_id = imp.split('\\')[-2]
                img_name = imp.split('\\')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path):
                    with open(img_save_path, 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                else:
                    img_data = cv2.imread(imp)
                    if crop_type == 'bbox':
                        b = list(map(int, b[0:4]))
                        cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                        img_features = img_pad(cropped_image)
                    elif 'surround' in crop_type:
                        b_org = list(map(int, b[0:4])).copy()
                        bbox = jitter_bbox(imp, [b], crop_resize_ratio)[0]
                        bbox = squarify(bbox, img_data.shape[1])
                        bbox = list(map(int, bbox[0:4]))
                        img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                        cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                        img_features = img_pad(cropped_image)
                    else:
                        raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    expanded_img = np.expand_dims(img_features, axis=0)
                    img_features = convolutional_net.predict(expanded_img)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
                img_seq.append(img_features)

            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        feat_shape = sequences.shape[1:]

        return sequences, feat_shape

    def get_data_sequence(self, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']
        d['speed'] = data_raw['obd_speed'].copy()

        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        backbone_name = ['vgg16']
        backbone_name = '_'.join(backbone_name).strip('_')
        eratio = model_opts['enlarge_ratio']

        data_gen_params = {'data_type': data_type, 'crop_type': 'bbox'}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        save_folder_name = '_'.join([feature_type, backbone_name])
        if 'surround' in feature_type:
            save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name, save_root_folder='data/features')
        return self.load_images_crop_and_process(data['image'], data['box_org'], data['ped_id'], **data_gen_params)

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated according
            to the length of optical flow window) and negative and positive sample counts
        """

        data_type_sizes_dict = {}
        data, neg_count, pos_count = self.get_data_sequence(data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses', save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    file_path=path_to_pose)
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts,
                       'train_opts': {'batch_size': batch_size, 'epochs': epochs, 'lr': lr}},
                      fid, default_flow_style=False)
        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count):
        """
        Computes class weights for imbalanced data used during training
        Args:
            apply_weights: Whether to apply weights
            sample_count: Positive and negative sample counts
        Returns:
            A dictionary of class weights or None if no weights to be calculated
        """
        if not apply_weights:
            return None

        total = sample_count['neg_count'] + sample_count['pos_count']

        # use simple ratio
        neg_weight = sample_count['pos_count'] / total
        pos_weight = sample_count['neg_count'] / total

        print("\n### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callbacks for training
        Args:
            learning_scheduler: Whether to use callbacks
            model_path: The path of model
        Returns:
            A list of call backs or None if learning_scheduler is false
        """
        callbacks = None

        # Set up learning schedulers
        if learning_scheduler:
            callbacks = []
            if 'early_stop' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'min_delta': 1.0, 'patience': 5,
                                  'verbose': 1}
                callbacks.append(EarlyStopping(**default_params))

            if 'plateau' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'factor': 0.2, 'patience': 5,
                                  'min_lr': 1e-08, 'verbose': 1}
                callbacks.append(ReduceLROnPlateau(**default_params))

            if 'checkpoint' in learning_scheduler:
                default_params = {'filepath': model_path, 'monitor': 'val_loss',
                                  'save_best_only': True, 'save_weights_only': False,
                                  'save_freq': 'epoch', 'verbose': 2}
                callbacks.append(ModelCheckpoint(**default_params))
        return callbacks

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              learning_scheduler=None,
              model_opts=None):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/'}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})
        data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']

        train_model = self.get_model(data_train['data_params'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = RMSprop(learning_rate=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)
        history = train_model.fit(x=data_train['data'][0],
                                  y=data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)

        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Graph
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, len(history.history["loss"])), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, len(history.history["val_loss"])), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, len(history.history["accuracy"])), history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, len(history.history["val_accuracy"])), history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        figure_path, _ = get_path(**path_params, file_name=("figureLR" + str(lr) + "Epoch" + str(epochs)
                                                            + "BatchSize" + str(batch_size) + ".png"))
        plt.savefig(figure_path, bbox_inches='tight')
        plt.close()

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    # Test Functions
    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)

        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})
        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)
        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        roc = roc_curve(test_data['data'][1], test_results)
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        pre_recall = precision_recall_curve(test_data['data'][1], test_results)

        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

    def get_model(self, data_params):
        """
        Generates a model
        Args:
            data_params: Data parameters to use for model generation
        Returns:
            A model
        """
        raise NotImplementedError("get_model should be implemented")

    # Auxiliary function
    def _gru(self, name='gru', r_state=False, r_sequence=False):
        """
        A helper function to create a single GRU unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the GRU
            r_sequence: Whether to return a sequence
        Return:
            A GRU unit
        """
        return GRU(units=self._num_hidden_units,
                   return_state=r_state,
                   return_sequences=r_sequence,
                   stateful=False,
                   kernel_regularizer=self._regularizationProcess,
                   recurrent_regularizer=self._regularizationProcess,
                   bias_regularizer=self._regularizationProcess,
                   name=name)


class MultiRNN(ActionPredict):
    """
    A multi-stream recurrent prediction model inspired by
    Bhattacharyya et al. "Long-term on-board prediction of people in traffic
    scenes under uncertainty." CVPR, 2018.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._rnn = self._gru
        self._rnn_cell = GRUCell

    def get_model(self, data_params):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i])(network_inputs[i]))

        if len(encoder_outputs) > 1:
            encodings = Concatenate(axis=1)(encoder_outputs)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense')(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        return net_model


class SFRNN(ActionPredict):
    """
    Pedestrian crossing prediction based on
    Rasouli et al. "Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs."
    BMVC, 2020. The original code can be found at https://github.com/aras62/SF-GRU
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # Network parameters
        self._rnn = self._gru
        self._rnn_cell = GRUCell

    def get_model(self, data_params):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        return_sequence = True
        num_layers = len(data_sizes)

        for i in range(num_layers):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

            if i == num_layers - 1:
                return_sequence = False

            if i == 0:
                x = self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i])
            else:
                x = Concatenate(axis=2)([x, network_inputs[i]])
                x = self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(x)

        model_output = Dense(1, activation='sigmoid', name='output_dense')(x)
        net_model = Model(inputs=network_inputs, outputs=model_output)

        return net_model


def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))

