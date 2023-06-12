from action_predict import action_prediction
from pie_data import PIE
import os
import yaml


def test_model(saved_folder_path=None):

    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as yaml_file:
        opts = yaml.safe_load(yaml_file)
    print(opts)
    model_opts = opts['model_opts']
    data_opts = opts['data_opts']
    net_opts = opts['net_opts']

    tte = model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte

    imdb = PIE(data_path="PIE-master")
    imdb.get_data_stats()

    method_class = action_prediction(model_opts['model'])(**net_opts)
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', data_opts['min_track_size'])
    acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)


if __name__ == '__main__':
    saved_files_path = "data/models/pie/MultiRNN/07Jun2023-10h14m06s"
    test_model(saved_folder_path=saved_files_path)
