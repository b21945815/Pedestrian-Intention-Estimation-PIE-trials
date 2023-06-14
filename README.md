# Pedestrian Crossing Action Prediction Benchmark

Benchmark for evaluating pedestrian action prediction algorithms that inlcude code for training, testing and evaluating baseline and state-of-the-art models for pedestrian action prediction on PIE and JAAD datasets.



**Kod buradan alındı: [https://github.com/ykotseruba/PedestrianActionBenchmark/tree/main]                                                                            
**Paper: [I. Kotseruba, A. Rasouli, J.K. Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf)** (see [citation](#citation) information below).

!!! hiperparametreler config_files içinde ama min_track_size ve time_to_event hiperparametrelerinden birini değişirseniz oluşturduğum veri setini tekrar oluşturmanız gerekir                                                                                                                                                       
                                                                                        
# Çalıştırma
1-) "dataPreparation.py" ile veriler elde edilir ama eksik dosyalar olduğundan dolayı bunu kullanmayınız. Verilerle beraber github'a yükledim.                                                                                           
2-) "train_test.py" run edilerek model çalıştırılır(bu hem train hem test yapıyor)       
3-) "test.py" dosyası ile model'in kaydedildiği adresi vererek test edebilirsiniz  
