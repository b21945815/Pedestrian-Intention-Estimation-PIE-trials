import time
import yaml
from dataUtilities import *
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRU, LSTM, GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell, RNN
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import resnet50
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


class ActionPredict(object):
    """
        A base interface class for prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._generator = None  # use data generator for train/test

    # Processing images anf generate features
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={}\
              \nsave_path={}, ".format(data_type, crop_type, save_path))

        preprocess_input = resnet50.preprocess_input

        convolutional_net = resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('\\')[-3]
                vid_id = imp.split('\\')[-2]
                img_name = imp.split('\\')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                if os.path.exists(img_save_path) and not regen_data:
                    if not self._generator:
                        with open(img_save_path, 'rb') as fid:
                            try:
                                img_features = pickle.load(fid)
                            except:
                                img_features = pickle.load(fid, encoding='bytes')
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    img_data = cv2.imread(imp)
                    if flip_image:
                        img_data = cv2.flip(img_data, 1)
                    if crop_type == 'bbox':
                        b = list(map(int, b[0:4]))
                        cropped_image = img_data[b[1]:b[3], b[0]:b[2], :]
                        img_features = img_pad(cropped_image, size=target_dim[0])
                    elif 'surround' in crop_type:
                        b_org = list(map(int, b[0:4])).copy()
                        bbox = jitter_bbox(imp, [b], crop_resize_ratio)[0]
                        bbox = squarify(bbox, img_data.shape[1])
                        bbox = list(map(int, bbox[0:4]))
                        img_data[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                        cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                        img_features = img_pad(cropped_image, size=target_dim[0])
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

                # if using the generator save the cached features path and size of the features
                if not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                if self._generator:
                    img_seq.append(img_save_path)
                else:
                    img_seq.append(img_features)
            sequences.append(img_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if self._global_pooling in ['max', 'avg']:
                feat_shape = feat_shape[-1]
            else:
                feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape

        # Processing images anf generate features

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
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

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']  # if data_type == 'train' else 0.0
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

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        backbone_name = ['resnet50']
        backbone_name = '_'.join(backbone_name).strip('_')
        eratio = model_opts['enlarge_ratio']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
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
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """

        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

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

        # make the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        else:
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
        # with open(config_path, 'wt') as fid:
        #     fid.write("####### Model options #######\n")
        #     for k in opts:
        #         fid.write("%s: %s\n" % (k, str(opts[k])))

        #     fid.write("\n####### Network config #######\n")
        #     # fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
        #     # fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))

        #     fid.write("\n####### Training config #######\n")
        #     fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
        #     fid.write("%s: %s\n" % ('epochs', str(epochs)))
        #     fid.write("%s: %s\n" % ('lr', str(lr)))

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
        # formula from sklearn
        # neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        # pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0

        # use simple ratio
        neg_weight = sample_count['pos_count'] / total
        pos_weight = sample_count['neg_count'] / total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callabcks for training
        Args:
            learning_scheduler: Whether to use callbacks
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
                default_params.update(learning_scheduler['early_stop'])
                callbacks.append(EarlyStopping(**default_params))

            if 'plateau' in learning_scheduler:
                default_params = {'monitor': 'val_loss',
                                  'factor': 0.2, 'patience': 5,
                                  'min_lr': 1e-08, 'verbose': 1}
                default_params.update(learning_scheduler['plateau'])
                callbacks.append(ReduceLROnPlateau(**default_params))

            if 'checkpoint' in learning_scheduler:
                default_params = {'filepath': model_path, 'monitor': 'val_loss',
                                  'save_best_only': True, 'save_weights_only': False,
                                  'save_freq': 'epoch', 'verbose': 2}
                default_params.update(learning_scheduler['checkpoint'])
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
            optimizer: Optimizer for training
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

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = RMSprop(learning_rate=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = self.get_callbacks(learning_scheduler, model_path)
        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

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
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

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

        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt': test_data['data'][1],
                         'y': test_results,
                         'image': test_data['image']}, picklefile)

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
                   kernel_regularizer=self._regularizer,
                   recurrent_regularizer=self._regularizer,
                   bias_regularizer=self._regularizer,
                   name=name)

    def _lstm(self, name='lstm', r_state=False, r_sequence=False):
        """
        A helper function to create a single LSTM unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the LSTM
            r_sequence: Whether to return a sequence
        Return:
            A LSTM unit
        """
        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    name=name)


class MultiRNN(ActionPredict):
    """
    A multi-stream recurrent prediction model inspired by
    Bhattacharyya et al. "Long-term on-board prediction of people in traffic
    scenes under uncertainty." CVPR, 2018.
    """

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

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

    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

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


class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 global_pooling=None,
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False):
        self.data = data
        self.labels = labels
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size else batch_size
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
        else:
            return X

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.global_pooling == 'max':
            img_features = np.squeeze(img_features)
            img_features = np.amax(img_features, axis=0)
            img_features = np.amax(img_features, axis=0)
        elif self.global_pooling == 'avg':
            img_features = np.squeeze(img_features)
            img_features = np.average(img_features, axis=0)
            img_features = np.average(img_features, axis=0)
        else:
            img_features = img_features.ravel()
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1] // len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            # for static model if only one image in the sequence
                            features_batch[i,] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i, ..., j * num_ch:j * num_ch + num_ch] = img_features
                            else:
                                features_batch[i, j,] = img_features
                else:
                    features_batch[i,] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        return np.array(self.labels[indices])
