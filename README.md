# Pedestrian Crossing Action Prediction Benchmark

Benchmark for evaluating pedestrian action prediction algorithms that inlcude code for training, testing and evaluating baseline and state-of-the-art models for pedestrian action prediction on PIE and JAAD datasets.



**Kod buradan alındı: https://github.com/ykotseruba/PedestrianActionBenchmark/tree/main 
**Paper: [I. Kotseruba, A. Rasouli, J.K. Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf)** (see [citation](#citation) information below).


# Eksik dosyalar
Videolar                                                                                                                                                           
PIE-MASTER içindeki data_cache klasörü                                                                                                                              
bu kodların alındığı yerdeki data klasörü içindeki models klasörü                                                                                                

# Çalıştırma
1-) not.py run edilerek data çekilir(sadece 1.video'lar olsun)                                                                                                      
"annotated" yazıyor ama asıl yapmamız gereken "all". Her şeyi hazırlayınca "all"'u çalıştarabilecek pc buldum                                                       2-) "train_test.py" run edilerek model çalıştırılır(bu hem train hem test yapıyor)                                                                                    3-) "test.py" dosyasını isterseniz deneyebilirsiniz(saved_files_path = sys.argv[1] bu kısım hata verebilir)                                                         4-) Sonra öncelikli olarak "pie_data.py" harici dosyalardaki kodları tekrar yazmaya odaklanmalıyız. Hatta bazı dosaları falan birleştiririz, fonksiyon adları ve yapı aynı olmasın diye vb.                                                                                                                                           Data fonksiyonları -Model ve train fonksiyonları -Test fonksiyonları, grafikler vb.                                                                             Herkes bu 3 parçadan birini yapsın diye düşünüyorum


## Authors

* **[Iuliia Kotseruba](http://www.cse.yorku.ca/~yulia_k/)**
* **[Amir Rasouli](http://www.cse.yorku.ca/~aras/index.html)**

Please email yulia_k@eecs.yorku.ca or arasouli.ai@gmail.com if you have any issues with running the code or using the data.

<a name="license"></a>
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


<a name="citation"></a>
## Citation

If you use the results, analysis or code for the models presented in the paper, please cite:

```
@inproceedings{kotseruba2021benchmark,
	title={{Benchmark for Evaluating Pedestrian Action Prediction}},
	author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
	booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
	pages={1258--1268},
	year={2021}
}
```

If you use model implementations provided in the benchmark, please cite the corresponding papers

- ATGC [1] 
- C3D [2]
- ConvLSTM [3]
- HierarchicalRNN [4]
- I3D [5]
- MultiRNN [6]
- PCPA [7]
- SFRNN [8] 
- SingleRNN [9]
- StackedRNN [10]
- Two_Stream [11]

[1] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Are they going to cross?  A benchmark dataset and baseline for pedestrian crosswalk behavior.  ICCVW, 2017.

[2] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,and Manohar Paluri. Learning spatiotemporal features with 3D convolutional networks. ICCV, 2015.

[3] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung,Wai-Kin Wong, and Wang-chun Woo. Convolutional LSTM network:  A machine learning approach for precipitation nowcasting. NeurIPS, 2015.

[4] Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. CVPR, 2015

[5] Joao Carreira and Andrew Zisserman.  Quo vadis, action recognition?  A new model and the kinetics dataset.  CVPR, 2017.

[6] Apratim Bhattacharyya, Mario Fritz, and Bernt Schiele. Long-term on-board prediction of people in traffic scenes under uncertainty. CVPR, 2018.

[7] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021.

[8] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs. BMVC, 2019

[9] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos.  Do They Want to Cross? Understanding Pedestrian Intention for Behavior Prediction. In IEEE Intelligent Vehicles Symposium (IV), 2020.

[10] Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vi-jayanarasimhan, Oriol Vinyals, Rajat Monga, and GeorgeToderici. Beyond short snippets: Deep networks for video classification. CVPR, 2015.

[11] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. NeurIPS, 2014.
