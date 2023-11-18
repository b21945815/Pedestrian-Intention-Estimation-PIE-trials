# Pedestrian Crossing Action Prediction Benchmark

Benchmark for evaluating pedestrian action prediction algorithms that inlcude code for training, testing and evaluating baseline and state-of-the-art models for pedestrian action prediction on PIE and JAAD datasets.



**Main source of codes: [https://github.com/ykotseruba/PedestrianActionBenchmark/tree/main]     
We worked on only one model here. Data is prepared with [https://github.com/aras62/PIE/blob/master/pie_data.py]                                                                                                                       
**Paper: [I. Kotseruba, A. Rasouli, J.K. Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf)** 

!!!hyperparameters are in config_files but if you change one of the hyperparameters min_track_size or time_to_event you have to re-make the dataset                                                                                                                                                  
                                                                                        
# Çalıştırma
1-) Data is obtained with "dataPreparation.py", but do not use it because there are missing files(videos etc.). I uploaded data to github.
2-) The model is run by running "train_test.py" (this does both train and test)
3-) You can test it by providing the address where the model is saved with the "test.py" file.
