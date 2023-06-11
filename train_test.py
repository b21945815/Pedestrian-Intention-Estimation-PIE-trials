import os
import yaml
from action_predict import action_prediction
from pie_data import PIE


def write_to_yaml(yaml_path=None, data=None):
    """
    Write model to yaml results file
    
    Args:
        yaml_path (None, optional): Description
        data (None, optional): results from the run
    
    Deleted Parameters:
        exp_type (str, optional): experiment type
        overwrite (bool, optional): whether to overwrite the results if the model exists
    """
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(data, yamlfile)


def run(config_file=None):
    """
    Run train and test on the dataset with parameters specified in configuration file.
    
    Args:
        config_file: path to configuration file in yaml format
    """
    print(config_file)
    # Read default Config file
    configs_default = 'config_files/configs_default.yaml'
    with open(configs_default, 'r') as f:
        configs = yaml.safe_load(f)

    with open(config_file, 'r') as f:
        model_configs = yaml.safe_load(f)

    # Update configs based on the model configs
    for k in ['model_opts', 'net_opts']:
        if k in model_configs:
            configs[k].update(model_configs[k])

    # Calculate min track size
    tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
        configs['model_opts']['time_to_event'][1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    # update model and training options from the config file
    for dataset_idx, dataset in enumerate(model_configs['exp_opts']['datasets']):
        configs['train_opts']['batch_size'] = model_configs['exp_opts']['batch_size'][dataset_idx]
        configs['train_opts']['lr'] = model_configs['exp_opts']['lr'][dataset_idx]
        configs['train_opts']['epochs'] = model_configs['exp_opts']['epochs'][dataset_idx]

        for k, v in configs.items():
            print(k, v)

        imdb = PIE(data_path="PIE-master")

        beh_seq_train = imdb.generate_data_trajectory_sequence('train', configs['data_opts']['min_track_size'])
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', configs['data_opts']['min_track_size'])
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', configs['data_opts']['min_track_size'])

        # get the model
        method_class = action_prediction(configs['model_opts']['model'])(**configs['net_opts'])
        # train and save the model
        saved_files_path = method_class.train(beh_seq_train, beh_seq_val, **configs['train_opts'],
                                              model_opts=configs['model_opts'])

        # test and evaluate the model
        acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)

        # save the results
        data = {}
        data['results'] = {}
        data['results']['acc'] = float(acc)
        data['results']['auc'] = float(auc)
        data['results']['f1'] = float(f1)
        data['results']['precision'] = float(precision)
        data['results']['recall'] = float(recall)
        write_to_yaml(yaml_path=os.path.join(saved_files_path, 'results.yaml'), data=data)

        data = configs
        write_to_yaml(yaml_path=os.path.join(saved_files_path, 'configs.yaml'), data=data)

        print('Model saved to {}'.format(saved_files_path))


if __name__ == '__main__':
    run(config_file="C:/Users/90553/Desktop/Kod/python/Group418/config_files/SFRNN.yaml")
    run(config_file="C:/Users/90553/Desktop/Kod/python/Group418/config_files/MultiRNN.yaml")

