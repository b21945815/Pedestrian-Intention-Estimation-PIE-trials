from action_predict import action_prediction
from pie_data import PIE
import os
import yaml
from train import write_to_yaml


def test_model(saved_folder_path=None):

    with open(os.path.join(saved_folder_path, 'configs.yaml'), 'r') as yaml_file:
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
    data = {}
    data['results'] = {}
    data['results']['acc'] = float(acc)
    data['results']['auc'] = float(auc)
    data['results']['f1'] = float(f1)
    data['results']['precision'] = float(precision)
    data['results']['recall'] = float(recall)
    write_to_yaml(yaml_path=os.path.join(saved_files_path, 'results.yaml'), data=data)


if __name__ == '__main__':
    saved_files_path = "data/models/pie/SFRNN/13Jun2023-11h59m55s"
    test_model(saved_folder_path=saved_files_path)
