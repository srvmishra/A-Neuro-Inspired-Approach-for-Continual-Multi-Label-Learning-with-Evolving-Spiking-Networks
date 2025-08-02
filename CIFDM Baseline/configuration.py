import numpy as np
import torch
class Config:
    def __init__(self, args):
#         self.data = args.data
#         self.log = args.log
        self.upper_bound = False
        self.name = args.name
        self.device = torch.device('cuda:{}'.format(args.device))
        if args.shuffle == 1:
            self.shuffle = True
        else:
            self.shuffle = False
        self.seed = args.seed
        self.generator = np.random.RandomState(self.seed)
        # use RandomState above instead of default_rng and done
        #-------------------------------------------------------
        # take these codes to baselines folder. done
        #-------------------------------------------------------
        # check the data split for consistency
        # run for the same 3 seeds on each dataset
        #-------------------------------------------------------
        # include all metrics. done.
        #--------------------------------------------------------
        # record only final individual and final combined evaluation modes
        # in the last sheet, compare these two modes for all datasets that
        # give meaningful results.

#         self.data_name = 'VOC'#'nuswide''yelp''yeast'
#         self.data_path = '../data/files/VOC2007/'
#         self.feature_path = '../data/VOCdevkit/VOC2007/features'
#         self.attri_num = 2048
#         self.embed_dim = 256
        # upper bound
#         self.label_list = [20, 0]
#         self.train_instance_list = [5011, 0]
        # setting 1
#         self.setting = 'setting 1'
#         self.label_list = [4, 4, 4, 4, 4]
#         self.train_instance_list = [974, 1171, 939, 978, 949]
        # setting 2
#         self.setting = 'setting 2'
#         self.label_list = [8, 3, 3, 3, 3]
#         self.train_instance_list = [1486, 798, 867, 1207, 653]
        # setting 3
#         self.setting = 'setting 3'
#         self.label_list = [12, 2, 2, 2, 2]
#         self.train_instance_list = [2017, 543, 1518, 454, 479]
#         self.task_num = len(self.label_list)
    
        if self.name == 'yeast':
            self.data_name = 'yeast'
            self.attri_num = 103
            # self.label_list = [7, 3, 3]
            # self.train_instance_list = [500, 500, 500]
            self.label_list = [2]*5 + [3]
            self.train_instance_list = [250]*6
        
        if self.name == 'emotions':
            self.data_name = 'emotions'
            self.attri_num = 72
            self.label_list = [2, 2, 2]
            self.train_instance_list = [130, 130, 131]
        
        if self.name == 'flags':
            self.data_name = 'flags'
            self.attri_num = 19
            self.label_list = [3, 2, 2]
            self.train_instance_list = [43, 43, 43]

        if self.name == 'gpositive':
            self.data_name = 'gpositive'
            self.attri_num = 440
            self.label_list = [2, 1, 1]
            self.train_instance_list = [105, 103, 103] #[103, 103, 105]

        if self.name == 'gnegative':
            self.data_name = 'gnegative'
            self.attri_num = 440
            self.label_list = [3, 2, 3]
            self.train_instance_list = [350, 350, 136] #[278, 280, 278] # # [278, 280, 278]

        if self.name == 'plant':
            self.data_name = 'plant'
            self.attri_num = 440
            self.label_list = [4, 4, 4]
            self.train_instance_list = [196, 196, 196]

        if self.name == 'virus':
            self.data_name = 'virus'
            self.attri_num = 440
            self.label_list = [2, 2, 2]
            self.train_instance_list = [32, 60, 32] #[41, 41, 42] # #[41, 41, 42]

        if self.name == 'human':
            self.data_name = 'human'
            self.attri_num = 440
            # self.label_list = [4, 6, 4]
            # self.train_instance_list = [621, 621, 620]
            self.label_list = [2]*4 + [3]
            self.train_instance_list = [310]*4 + [622]

        if self.name == 'eukaryote':
            self.data_name = 'eukaryote'
            self.attri_num = 440
            # self.label_list = [8, 7, 7]
            # self.train_instance_list = [1552, 1552, 1554]
            self.label_list = [2]*9 + [1]
            self.train_instance_list = [435, 530, 438] + [465]*7

        if self.name == 'scene':
            self.data_name = 'scene'
            self.attri_num = 294
            self.label_list = [2, 2, 2]
            self.train_instance_list = [405, 403, 403]

        if self.name == 'foodtruck':
            self.data_name = 'foodtruck'
            self.attri_num = 21
            self.label_list = [4, 4, 4]
            self.train_instance_list = [90, 80, 80]

        if self.name == 'yeast_big':
            self.data_name = 'yeast'
            self.attri_num = 103
            self.label_list = [2, 2, 2, 2, 2, 3]
            self.train_instance_list = [250, 250, 250, 250, 250, 250]

        if self.name == 'foodtruck_big':
            self.data_name = 'foodtruck'
            self.attri_num = 21
            self.label_list = [2, 2, 2, 2, 2, 2]
            self.train_instance_list = [40, 40, 40, 40, 40, 50]

        if self.name == 'birds':
            self.data_name = 'birds'
            self.attri_num = 260
            self.label_list = [3]*5 + [4]
            self.train_instance_list = [54]*5 + [52]

        if self.name == 'enron':
            self.data_name = 'enron'
            self.attri_num = 1001
            self.label_list = [5]*9 + [8]
            self.train_instance_list = [112]*9 + [115]

        if self.name == 'CAL500':
            self.data_name = 'CAL500'
            self.attri_num = 68
            self.label_list = [9]*19 + [2]
            self.train_instance_list = [15]*20

        if self.name == 'medical':
            self.data_name = 'medical'
            self.attri_num = 1449
            self.label_list = [4]*9 + [9]
            self.train_instance_list = [33]*9 + [36]

        # if self.name == 'CAL500':
        #     self.data_name = 'CAL500'
        #     self.attri_num = 68
        #     self.label_list = [17]*9 + [21]
        #     self.train_instance_list = [25]*9 + [26]
            
        if self.name == 'mediamill_20':
            self.data_name = 'mediamill_20'
            self.attri_num = 120
            self.label_list = [5]*19 + [6]
            self.train_instance_list = [1549]*19 + [1562]

        if self.name == 'mediamill_50':
            self.data_name = 'mediamill_50'
            self.attri_num = 120
            self.label_list = [2]*49 + [3]
            self.train_instance_list = [619]*49 + [662]

        if self.name == 'delicious_20':
            self.data_name = 'delicious_20'
            self.attri_num = 500
            self.label_list = [49]*19 + [52]
            self.train_instance_list = [646]*20

        if self.name == 'delicious_50':
            self.data_name = 'delicious_50'
            self.attri_num = 500
            self.label_list = [20]*49 + [3]
            self.train_instance_list = [258]*49 + [278]        
        
        self.task_num = len(self.label_list)
        self.embed_dim = 20 #200
    
    
#         self.data_name = 'COCO'#'nuswide''yelp''yeast'
#         self.data_path = '../data/files/COCO2014/'
#         self.feature_path = '../data/COCO/features'
#         self.attri_num = 2048
#         self.embed_dim = 128
        # upper bound
#         self.label_list = [80, 0]
#         self.train_instance_list = [82081, 0]
        # setting 1
#         self.setting = 'setting 1'
#         self.label_list = [16, 16, 16, 16, 16]
#         self.train_instance_list = [17536, 14779, 16656, 15748, 17362]
        # setting 2
#         self.setting = 'setting 2'
#         self.label_list = [20, 15, 15, 15, 15]
#         self.train_instance_list = [16737, 15772, 16508, 16715, 16349]
        # setting 3
#         self.setting = 'setting 3'
#         self.label_list = [40, 10, 10, 10, 10]
#         self.train_instance_list = [17387, 18922, 16489, 13667, 15616]
        self.task_num = len(self.label_list)

#         self.first_epoch = 40
        self.first_epoch = 20
        self.pse_epoch = 1
        self.ssl_epoch = 1
#         self.new_epoch = 40
        self.new_epoch = 20
        self.sti_epoch = 15
#         self.ste_epoch = 40
        self.ste_epoch = 20
        self.ts_epoch = 5
        self.as_epoch = 20

        self.first_batch = 64
        self.ssl_batch = 64
        self.new_batch = 64
        self.st_batch = 64
        self.ts_batch = 64
        self.as_batch = 64
        self.eval_batch = 256

        self.num_workers = 24
        self.weight = 30
        self.use_teacher = False
        self.gamma = 8

    def __str__(self):
        result = 'data_name: {}.\nseed: {}.\nshuffle: {}.\nupper_bound: {}.\nembed_dim: {}.\ntask_num: {}.\nweight: {}.\nuse_teacher: {}.\ngamma: {}.\n'.format(self.data_name, self.seed, self.shuffle, self.upper_bound, self.embed_dim, self.task_num, self.weight, self.use_teacher, self.gamma)
        result += 'first_epoch: {}.\npse_epoch: {}.\nssl_epoch {}.\nnew_epoch: {}.\nsti_epoch: {}.\nste_epoch: {}.\nts_epoch: {}.\nas_epoch: {}\n'.format(self.first_epoch, self.pse_epoch, self.ssl_epoch, self.new_epoch, self.sti_epoch, self.ste_epoch, self.ts_epoch, self.as_epoch)
        result += 'first_batch: {}.\nssl_batch: {}.\nnew_batch: {}.\nst_batch: {}.\nts_batch: {}.\nas_batch: {}\n'.format(self.first_batch, self.ssl_batch, self.new_batch, self.st_batch, self.ts_batch, self.as_batch)
        return  result


