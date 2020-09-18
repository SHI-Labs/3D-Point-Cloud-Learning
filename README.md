# Survey-on-3D-Point-Clouds

## Survey
Deep Learning for 3D Point Cloud Understanding: A Survey [ArXiv link](http://arxiv.org/)

## Datasets

- [ShapeNet](https://www.shapenet.org/)
- [ModelNet](https://modelnet.cs.princeton.edu/)
- [S3DIS](http://3dsemantics.stanford.edu/)
- [Semantic3D](http://www.semantic3d.net/)
- [ScanNet](http://www.scan-net.org/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [3DMatch](https://3dmatch.cs.princeton.edu/)
- [nuScenes](https://www.nuscenes.org/)
- [Lyft Level 5](https://self-driving.lyft.com/level5/data/)
- [Waymo Open Dataset](https://waymo.com/open/)

## Papers

### 3D Object Classification

#### Projection-based methods

- Su, Hang, et al. "Multi-view convolutional neural networks for 3d shape recognition." *Proceedings of the IEEE international conference on computer vision*. 2015. [[paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf)]

- Feng, Yifan, et al. "Gvcnn: Group-view convolutional neural networks for 3d shape recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf)]

- Yu, Tan, Jingjing Meng, and Junsong Yuan. "Multi-view harmonized bilinear network for 3d object recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Multi-View_Harmonized_Bilinear_CVPR_2018_paper.pdf)]

- Yang, Ze, and Liwei Wang. "Learning relationships for multi-view 3D object recognition." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Learning_Relationships_for_Multi-View_3D_Object_Recognition_ICCV_2019_paper.pdf)]

- Maturana, Daniel, and Sebastian Scherer. "Voxnet: A 3d convolutional neural network for real-time object recognition." *2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2015. [[paper](http://graphics.stanford.edu/courses/cs233-20-spring/ReferencedPapers/voxnet_07353481.pdf)]

- Chang, Angel X., et al. "Shapenet: An information-rich 3d model repository." *arXiv preprint arXiv:1512.03012* (2015). [[paper](https://arxiv.org/pdf/1512.03012.pdf)]

- Riegler, Gernot, Ali Osman Ulusoy, and Andreas Geiger. "Octnet: Learning deep 3d representations at high resolutions." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.pdf)]

- Prokudin, Sergey, Christoph Lassner, and Javier Romero. "Efficient learning on point clouds with basis point sets." *Proceedings of the IEEE International Conference on Computer Vision Workshops*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CEFRL/Prokudin_Efficient_Learning_on_Point_Clouds_with_Basis_Point_Sets_ICCVW_2019_paper.pdf)]

#### Point-based methods

- Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)]

- Qi, Charles Ruizhongtai, et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." *Advances in neural information processing systems*. 2017. [[paper](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf)]

- Zhao, Hengshuang, et al. "PointWeb: Enhancing local neighborhood features for point cloud processing." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_PointWeb_Enhancing_Local_Neighborhood_Features_for_Point_Cloud_Processing_CVPR_2019_paper.pdf)]

- Duan, Yueqi, et al. "Structural relational reasoning of point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Duan_Structural_Relational_Reasoning_of_Point_Clouds_CVPR_2019_paper.pdf)]

- Komarichev, Artem, Zichun Zhong, and Jing Hua. "A-CNN: Annularly convolutional neural networks on point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Komarichev_A-CNN_Annularly_Convolutional_Neural_Networks_on_Point_Clouds_CVPR_2019_paper.pdf)]

- Liu, Yongcheng, et al. "Relation-shape convolutional neural network for point cloud analysis." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Relation-Shape_Convolutional_Neural_Network_for_Point_Cloud_Analysis_CVPR_2019_paper.pdf)]

- Wu, Wenxuan, Zhongang Qi, and Li Fuxin. "Pointconv: Deep convolutional networks on 3d point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_PointConv_Deep_Convolutional_Networks_on_3D_Point_Clouds_CVPR_2019_paper.pdf)]

- Hermosilla, Pedro, et al. "Monte carlo convolution for learning on non-uniformly sampled point clouds." *ACM Transactions on Graphics (TOG)* 37.6 (2018): 1-12. [[paper](https://dl.acm.org/doi/pdf/10.1145/3272127.3275110)]

- Lan, Shiyi, et al. "Modeling local geometric structure of 3D point clouds using Geo-CNN." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lan_Modeling_Local_Geometric_Structure_of_3D_Point_Clouds_Using_Geo-CNN_CVPR_2019_paper.pdf)]
- Rao, Yongming, Jiwen Lu, and Jie Zhou. "Spherical fractal convolutional neural networks for point cloud recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Rao_Spherical_Fractal_Convolutional_Neural_Networks_for_Point_Cloud_Recognition_CVPR_2019_paper.pdf)]

- Simonovsky, Martin, and Nikos Komodakis. "Dynamic edge-conditioned filters in convolutional neural networks on graphs." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.pdf)]

- Wang, Yue, et al. "Dynamic graph cnn for learning on point clouds." *Acm Transactions On Graphics (tog)* 38.5 (2019): 1-12. [[paper](https://arxiv.org/pdf/1801.07829.pdf)]
- Hassani, Kaveh, and Mike Haley. "Unsupervised multi-task feature learning on point clouds." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hassani_Unsupervised_Multi-Task_Feature_Learning_on_Point_Clouds_ICCV_2019_paper.pdf)]
- Chen, Chao, et al. "Clusternet: Deep hierarchical cluster network with rigorously rotation-invariant representation for point cloud analysis." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_ClusterNet_Deep_Hierarchical_Cluster_Network_With_Rigorously_Rotation-Invariant_Representation_for_CVPR_2019_paper.pdf)]

- Klokov, Roman, and Victor Lempitsky. "Escape from cells: Deep kd-networks for the recognition of 3d point cloud models." *Proceedings of the IEEE International Conference on Computer Vision*. 2017. [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf)]

- Zeng, Wei, and Theo Gevers. "3DContextNet: Kd tree guided hierarchical learning of point clouds using local and global contextual cues." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018. [[paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Zeng_3DContextNet_K-d_Tree_Guided_Hierarchical_Learning_of_Point_Clouds_Using_ECCVW_2018_paper.pdf)]
- Wu, Pengxiang, et al. "Point cloud processing via recurrent set encoding." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 33. 2019. [[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4484/4362)]

- Li, Jiaxin, Ben M. Chen, and Gim Hee Lee. "So-net: Self-organizing network for point cloud analysis." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)]

### 3D Segmentation

#### Semantic Segmentation

- Huang, Jing, and Suya You. "Point cloud labeling using 3d convolutional neural network." *2016 23rd International Conference on Pattern Recognition (ICPR)*. IEEE, 2016. [[paper](https://www.researchgate.net/profile/Jing_Huang37/publication/308349377_Point_Cloud_Labeling_using_3D_Convolutional_Neural_Network/links/5d60108592851c619d71a3a7/Point-Cloud-Labeling-using-3D-Convolutional-Neural-Network.pdf)]
- Dai, Angela, et al. "Scancomplete: Large-scale scene completion and semantic segmentation for 3d scans." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dai_ScanComplete_Large-Scale_Scene_CVPR_2018_paper.pdf)]

- Meng, Hsien-Yu, et al. "Vv-net: Voxel vae net with group convolutions for point cloud segmentation." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Meng_VV-Net_Voxel_VAE_Net_With_Group_Convolutions_for_Point_Cloud_ICCV_2019_paper.pdf)]
- Lawin, Felix Järemo, et al. "Deep projective 3D semantic segmentation." *International Conference on Computer Analysis of Images and Patterns*. Springer, Cham, 2017. [[paper](https://arxiv.org/pdf/1705.03428.pdf)]

- Zhang, Yang, et al. "PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_PolarNet_An_Improved_Grid_Representation_for_Online_LiDAR_Point_Clouds_CVPR_2020_paper.pdf)]
- Dai, Angela, and Matthias Nießner. "3dmv: Joint 3d-multi-view prediction for 3d semantic scene segmentation." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018. [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Angela_Dai_3DMV_Joint_3D-Multi-View_ECCV_2018_paper.pdf)]

- Jaritz, Maximilian, Jiayuan Gu, and Hao Su. "Multi-view pointnet for 3d scene understanding." *Proceedings of the IEEE International Conference on Computer Vision Workshops*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/GMDL/Jaritz_Multi-View_PointNet_for_3D_Scene_Understanding_ICCVW_2019_paper.pdf)]

- Engelmann, Francis, et al. "Know what your neighbors do: 3D semantic segmentation of point clouds." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018. [[paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Engelmann_Know_What_Your_Neighbors_Do_3D_Semantic_Segmentation_of_Point_ECCVW_2018_paper.pdf)]

- Wang, Shenlong, et al. "Deep parametric continuous convolutional neural networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf)]

- Liu, Zhijian, et al. "Point-Voxel CNN for efficient 3D deep learning." *Advances in Neural Information Processing Systems*. 2019. [[paper](https://papers.nips.cc/paper/8382-point-voxel-cnn-for-efficient-3d-deep-learning.pdf)]

- Hua, Binh-Son, Minh-Khoi Tran, and Sai-Kit Yeung. "Pointwise convolutional neural networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.pdf)]

- Landrieu, Loic, and Martin Simonovsky. "Large-scale point cloud semantic segmentation with superpoint graphs." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf)]
- Landrieu, Loic, and Mohamed Boussaha. "Point cloud oversegmentation with graph-structured deep metric learning." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Landrieu_Point_Cloud_Oversegmentation_With_Graph-Structured_Deep_Metric_Learning_CVPR_2019_paper.pdf)]

- Wang, Lei, et al. "Graph attention convolution for point cloud semantic segmentation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Graph_Attention_Convolution_for_Point_Cloud_Semantic_Segmentation_CVPR_2019_paper.pdf)]

- Tatarchenko, Maxim, et al. "Tangent convolutions for dense prediction in 3d." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tatarchenko_Tangent_Convolutions_for_CVPR_2018_paper.pdf)]
- Hu, Qingyong, et al. "RandLA-Net: Efficient semantic segmentation of large-scale point clouds." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf)]
- Xu, Xun, and Gim Hee Lee. "Weakly Supervised Semantic Point Cloud Segmentation: Towards 10x Fewer Labels." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Weakly_Supervised_Semantic_Point_Cloud_Segmentation_Towards_10x_Fewer_Labels_CVPR_2020_paper.pdf)]
- Wei, Jiacheng, et al. "Multi-Path Region Mining For Weakly Supervised 3D Semantic Segmentation on Point Clouds." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Path_Region_Mining_for_Weakly_Supervised_3D_Semantic_Segmentation_on_CVPR_2020_paper.pdf)]

#### Instance Segmentation

- Hou, Ji, Angela Dai, and Matthias Nießner. "3d-sis: 3d semantic instance segmentation of rgb-d scans." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf)]
- Yi, Li, et al. "Gspn: Generative shape proposal network for 3d instance segmentation in point cloud." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_GSPN_Generative_Shape_Proposal_Network_for_3D_Instance_Segmentation_in_CVPR_2019_paper.pdf)]

- Yang, Ze, and Liwei Wang. "Learning relationships for multi-view 3D object recognition." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Learning_Relationships_for_Multi-View_3D_Object_Recognition_ICCV_2019_paper.pdf)]
- Zhang, Feihu, et al. "Instance segmentation of lidar point clouds." *ICRA, Cited by* 4.1 (2020). [[paper](http://www.feihuzhang.com/ICRA2020.pdf)]
- Wang, Weiyue, et al. "Sgpn: Similarity group proposal network for 3d point cloud instance segmentation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)]
- Lahoud, Jean, et al. "3d instance segmentation via multi-task metric learning." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)]
- Zhang, Biao, and Peter Wonka. "Point cloud instance segmentation using probabilistic embeddings." *arXiv preprint arXiv:1912.00145* (2019). [[paper](https://arxiv.org/pdf/1912.00145.pdf)]
- Wu, Bichen, et al. "Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud." *2018 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2018. [[paper](https://arxiv.org/pdf/1710.07368.pdf)]
- Wu, Bichen, et al. "Squeezesegv2: Improved model structure and unsupervised domain adaptation for road-object segmentation from a lidar point cloud." *2019 International Conference on Robotics and Automation (ICRA)*. IEEE, 2019. [[paper](https://arxiv.org/pdf/1809.08495.pdf)]
- Lyu, Yecheng, Xinming Huang, and Ziming Zhang. "Learning to Segment 3D Point Clouds in 2D Image Space." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lyu_Learning_to_Segment_3D_Point_Clouds_in_2D_Image_Space_CVPR_2020_paper.pdf)]
- Jiang, Li, et al. "PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.pdf)]

#### Joint Training

- Hassani, Kaveh, and Mike Haley. "Unsupervised multi-task feature learning on point clouds." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hassani_Unsupervised_Multi-Task_Feature_Learning_on_Point_Clouds_ICCV_2019_paper.pdf)]
- Pham, Quang-Hieu, et al. "JSIS3D: joint semantic-instance segmentation of 3d point clouds with multi-task pointwise networks and multi-value conditional random fields." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pham_JSIS3D_Joint_Semantic-Instance_Segmentation_of_3D_Point_Clouds_With_Multi-Task_CVPR_2019_paper.pdf)]
- Wang, Xinlong, et al. "Associatively segmenting instances and semantics in point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Associatively_Segmenting_Instances_and_Semantics_in_Point_Clouds_CVPR_2019_paper.pdf)]

### 3D Object Detection

#### Projection-based methods

- Zhou, Yin, and Oncel Tuzel. "Voxelnet: End-to-end learning for point cloud based 3d object detection." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf)]
- Yan, Yan, Yuxing Mao, and Bo Li. "Second: Sparsely embedded convolutional detection." *Sensors* 18.10 (2018): 3337. [[paper](https://www.mdpi.com/1424-8220/18/10/3337/htm)]
- Lang, Alex H., et al. "Pointpillars: Fast encoders for object detection from point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)]
- Wang, Yue, et al. "Pillar-based Object Detection for Autonomous Driving." *arXiv preprint arXiv:2007.10323* (2020). [[paper](https://arxiv.org/pdf/2007.10323.pdf)]
- He, Chenhang, et al. "Structure Aware Single-stage 3D Object Detection from Point Cloud." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf)]

#### Point-based methods

- Yang, Zetong, et al. "Std: Sparse-to-dense 3d object detector for point cloud." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_STD_Sparse-to-Dense_3D_Object_Detector_for_Point_Cloud_ICCV_2019_paper.pdf)]
- Shi, Shaoshuai, Xiaogang Wang, and Hongsheng Li. "Pointrcnn: 3d object proposal generation and detection from point cloud." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf)]
- Qi, Charles R., et al. "Deep hough voting for 3d object detection in point clouds." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_Deep_Hough_Voting_for_3D_Object_Detection_in_Point_Clouds_ICCV_2019_paper.pdf)]
- Qi, Charles R., et al. "Imvotenet: Boosting 3d object detection in point clouds with image votes." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_ImVoteNet_Boosting_3D_Object_Detection_in_Point_Clouds_With_Image_CVPR_2020_paper.pdf)]
- Du, Liang, et al. "Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Du_Associate-3Ddet_Perceptual-to-Conceptual_Association_for_3D_Point_Cloud_Object_Detection_CVPR_2020_paper.pdf)]
- Yang, Zetong, et al. "3dssd: Point-based 3d single stage object detector." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_3DSSD_Point-Based_3D_Single_Stage_Object_Detector_CVPR_2020_paper.pdf)]
- Zarzar, Jesus, Silvio Giancola, and Bernard Ghanem. "PointRGCN: Graph convolution networks for 3D vehicles detection refinement." *arXiv preprint arXiv:1911.12236* (2019). [[paper](https://arxiv.org/pdf/1911.12236.pdf)]
- Chen, Jintai, et al. "A Hierarchical Graph Network for 3D Object Detection on Point Clouds." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Hierarchical_Graph_Network_for_3D_Object_Detection_on_Point_CVPR_2020_paper.pdf)]
- Shi, Weijing, and Raj Rajkumar. "Point-gnn: Graph neural network for 3d object detection in a point cloud." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf)]

#### Multi-view Fusion

- Chen, Xiaozhi, et al. "Multi-view 3d object detection network for autonomous driving." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)]

- Liang, Ming, et al. "Deep continuous fusion for multi-sensor 3d object detection." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018. [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)]

- Lu, Haihua, et al. "SCANet: Spatial-channel attention network for 3D object detection." *ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2019. [[paper](https://ieeexplore.ieee.org/abstract/document/8682746/)]
- Zeng, Yiming, et al. "Rt3d: Real-time 3-d vehicle detection in lidar point cloud for autonomous driving." *IEEE Robotics and Automation Letters* 3.4 (2018): 3434-3440. [[paper](https://meridiancas.github.io/publications/pdffiles/2018_RAL_RT3D.pdf)]

- Qi, Charles R., et al. "Frustum pointnets for 3d object detection from rgb-d data." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)]
- Gupta, Ayush. "Deep Sensor Fusion for 3D Bounding Box Estimation and Recognition of Objects." [[paper](http://cs230.stanford.edu/files_winter_2018/projects/6939556.pdf)]

### 3D Object Tracking

- Giancola, Silvio, Jesus Zarzar, and Bernard Ghanem. "Leveraging shape completion for 3d siamese tracking." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Giancola_Leveraging_Shape_Completion_for_3D_Siamese_Tracking_CVPR_2019_paper.pdf)]
- Zarzar, Jesus, Silvio Giancola, and Bernard Ghanem. "PointRGCN: Graph convolution networks for 3D vehicles detection refinement." *arXiv preprint arXiv:1911.12236* (2019). [[paper](https://arxiv.org/pdf/1911.12236.pdf)]
- Chiu, Hsu-kuang, et al. "Probabilistic 3d multi-object tracking for autonomous driving." *arXiv preprint arXiv:2001.05673* (2020). [[paper](https://arxiv.org/pdf/2001.05673.pdf)]
- Qi, Haozhe, et al. "P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_P2B_Point-to-Box_Network_for_3D_Object_Tracking_in_Point_Clouds_CVPR_2020_paper.pdf)]

### 3D Scene Flow Estimation

- Liu, Xingyu, Charles R. Qi, and Leonidas J. Guibas. "Flownet3d: Learning scene flow in 3d point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_FlowNet3D_Learning_Scene_Flow_in_3D_Point_Clouds_CVPR_2019_paper.pdf)]
- Wang, Zirui, et al. "FlowNet3D++: Geometric losses for deep scene flow estimation." *The IEEE Winter Conference on Applications of Computer Vision*. 2020. [[paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_FlowNet3D_Geometric_Losses_For_Deep_Scene_Flow_Estimation_WACV_2020_paper.pdf)]
- Gu, Xiuye, et al. "Hplflownet: Hierarchical permutohedral lattice flownet for scene flow estimation on large-scale point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gu_HPLFlowNet_Hierarchical_Permutohedral_Lattice_FlowNet_for_Scene_Flow_Estimation_on_CVPR_2019_paper.pdf)]
- Liu, Xingyu, Mengyuan Yan, and Jeannette Bohg. "Meteornet: Deep learning on dynamic 3d point cloud sequences." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_MeteorNet_Deep_Learning_on_Dynamic_3D_Point_Cloud_Sequences_ICCV_2019_paper.pdf)]
- Fan, Hehe, and Yi Yang. "PointRNN: Point recurrent neural network for moving point cloud processing." *arXiv preprint arXiv:1910.08287* (2019). [[paper](https://arxiv.org/pdf/1910.08287.pdf)]

### 3D Point Registration and Matching

- Lu, Weixin, et al. "Deepvcp: An end-to-end deep neural network for point cloud registration." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lu_DeepVCP_An_End-to-End_Deep_Neural_Network_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)]
- Gojcic, Zan, et al. "The perfect match: 3d point cloud matching with smoothed densities." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gojcic_The_Perfect_Match_3D_Point_Cloud_Matching_With_Smoothed_Densities_CVPR_2019_paper.pdf)]
- Gojcic, Zan, et al. "Learning multiview 3D point cloud registration." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gojcic_Learning_Multiview_3D_Point_Cloud_Registration_CVPR_2020_paper.pdf)]

- Yew, Zi Jian, and Gim Hee Lee. "RPM-Net: Robust Point Matching using Learned Features." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.pdf)]

### Augmentation and Completion

#### Discriminative methods

- Rakotosaona, Marie‐Julie, et al. "Pointcleannet: Learning to denoise and remove outliers from dense point clouds." *Computer Graphics Forum*. Vol. 39. No. 1. 2020. [[paper](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.13753)]

- Guerrero, Paul, et al. "Pcpnet learning local shape properties from raw point clouds." *Computer Graphics Forum*. Vol. 37. No. 2. 2018. [[paper](https://arxiv.org/pdf/1710.04954.pdf)]
- Hermosilla, Pedro, Tobias Ritschel, and Timo Ropinski. "Total Denoising: Unsupervised learning of 3D point cloud cleaning." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hermosilla_Total_Denoising_Unsupervised_Learning_of_3D_Point_Cloud_Cleaning_ICCV_2019_paper.pdf)]
- Nezhadarya, Ehsan, et al. "Adaptive Hierarchical Down-Sampling for Point Cloud Classification." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nezhadarya_Adaptive_Hierarchical_Down-Sampling_for_Point_Cloud_Classification_CVPR_2020_paper.pdf)]
- Lang, Itai, Asaf Manor, and Shai Avidan. "SampleNet: Differentiable Point Cloud Sampling." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf)]

#### Generative methods

- Xiang, Chong, Charles R. Qi, and Bo Li. "Generating 3d adversarial point clouds." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiang_Generating_3D_Adversarial_Point_Clouds_CVPR_2019_paper.pdf)]
- Shu, Dong Wook, Sung Woo Park, and Junseok Kwon. "3d point cloud generative adversarial network based on tree structured graph convolutions." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shu_3D_Point_Cloud_Generative_Adversarial_Network_Based_on_Tree_Structured_ICCV_2019_paper.pdf)]
- Zhou, Hang, et al. "DUP-Net: Denoiser and upsampler network for 3D adversarial point clouds defense." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_DUP-Net_Denoiser_and_Upsampler_Network_for_3D_Adversarial_Point_Clouds_ICCV_2019_paper.pdf)]
- Yu, Lequan, et al. "Pu-net: Point cloud upsampling network." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.pdf)]

- Yifan, Wang, et al. "Patch-based progressive 3d point set upsampling." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yifan_Patch-Based_Progressive_3D_Point_Set_Upsampling_CVPR_2019_paper.pdf)]
- Hui, Le, et al. "Progressive Point Cloud Deconvolution Generation Network." *arXiv preprint arXiv:2007.05361* (2020). [[paper](https://arxiv.org/pdf/2007.05361.pdf)]
- Yuan, Wentao, et al. "Pcn: Point completion network." *2018 International Conference on 3D Vision (3DV)*. IEEE, 2018. [[paper](https://arxiv.org/pdf/1808.00671.pdf)]
- Wang, Xiaogang, Marcelo H. Ang Jr, and Gim Hee Lee. "Cascaded Refinement Network for Point Cloud Completion." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cascaded_Refinement_Network_for_Point_Cloud_Completion_CVPR_2020_paper.pdf)]
- Huang, Zitian, et al. "PF-Net: Point Fractal Network for 3D Point Cloud Completion." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf)]
- Xie, Haozhe, et al. "GRNet: Gridding Residual Network for Dense Point Cloud Completion." *arXiv preprint arXiv:2006.03761* (2020). [[paper](https://arxiv.org/pdf/2006.03761.pdf)]
- Lan, Ziquan, Zi Jian Yew, and Gim Hee Lee. "Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lan_Robust_Point_Cloud_Based_Reconstruction_of_Large-Scale_Outdoor_Scenes_CVPR_2019_paper.pdf)]
- Li, Ruihui, et al. "Pu-gan: a point cloud upsampling adversarial network." *Proceedings of the IEEE International Conference on Computer Vision*. 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_PU-GAN_A_Point_Cloud_Upsampling_Adversarial_Network_ICCV_2019_paper.pdf)]
- Sarmad, Muhammad, Hyunjoo Jenny Lee, and Young Min Kim. "Rl-gan-net: A reinforcement learning agent controlled gan network for real-time point cloud shape completion." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sarmad_RL-GAN-Net_A_Reinforcement_Learning_Agent_Controlled_GAN_Network_for_Real-Time_CVPR_2019_paper.pdf)]

