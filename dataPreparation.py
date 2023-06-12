import os
from pie_data import PIE
import yaml
import shutil
from action_predict import action_prediction


def delete_folder(address):
    try:
        shutil.rmtree(address)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


pie_path = "PIE-master"
configs_default = 'config_files/configs_default.yaml'

with open(configs_default, 'r') as f:
    configs = yaml.safe_load(f)

tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
    configs['model_opts']['time_to_event'][1]
min_track_size = configs['model_opts']['obs_length'] + tte

videoSet = {'set01': ['video_0001', 'video_0002', 'video_0003', 'video_0004'],
            'set02': ['video_0001', 'video_0002', 'video_0003'],
            'set03': ['video_0001', 'video_0002', 'video_0003', 'video_0004', 'video_0005', 'video_0006',
                      'video_0007', 'video_0008', 'video_0009', 'video_0010', 'video_0011', 'video_0012',
                      'video_0013', 'video_0014', 'video_0015', 'video_0016', 'video_0017', 'video_0018', 'video_0019'],
            'set04': ['video_0001', 'video_0002', 'video_0003', 'video_0004', 'video_0005', 'video_0006',
                      'video_0007', 'video_0008', 'video_0009', 'video_0010', 'video_0011', 'video_0012',
                      'video_0013', 'video_0014', 'video_0015', 'video_0016'],
            'set05': ['video_0001', 'video_0002'],
            'set06': ['video_0001', 'video_0002', 'video_0003', 'video_0004', 'video_0005', 'video_0006',
                      'video_0007', 'video_0008', 'video_0009']}

setList = ['set01', 'set02', 'set03', 'set04', 'set05', 'set06']

imdb = PIE(data_path=pie_path)

method_class = action_prediction("MultiRNN")(**configs['net_opts'])
for setName in setList:
    if setName in ['set01', 'set02', 'set04']:
        set_type = 'train'
    elif setName in ['set05', 'set06']:
        set_type = 'val'
    else:
        set_type = 'test'
    videoList = videoSet[setName]
    for videoName in videoList:
        imdb.extract_and_save_images(setName, videoName)
        dataSequence = imdb.generate_data_trajectory_sequence(set_type, min_track_size, setName, videoName)
        dataInput, _, _ = method_class.get_data_sequence(dataSequence, configs['model_opts'])
        method_class.get_context_data(configs['model_opts'], dataInput, set_type, 'local_box')
        method_class.get_context_data(configs['model_opts'], dataInput, set_type, 'local_surround')
        delete_folder(
            "C://Users//90553//Desktop//Kod//python//Group418//PIE-master//images//" + setName + "//" + videoName)
        os.remove(
            "C://Users//90553//Desktop//Kod//python//Group418//PIE-master//PIE_clips//"
            + setName + "//" + videoName + ".mp4")
