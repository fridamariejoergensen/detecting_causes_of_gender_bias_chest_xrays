import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.transforms.functional as F
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import os
import random

DISEASE_LABELS_CHE = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture']

class ChexpertDatasetNew(Dataset):
    def __init__(self, img_data_dir, df_data, image_size, augmentation=False, pseudo_rgb = False,single_label=None,
         disease_labels_list=DISEASE_LABELS_CHE):
        self.img_data_dir = img_data_dir
        self.df_data = df_data
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.single_label = single_label

        self.labels=disease_labels_list
        if self.single_label is not None:
            self.labels = [self.single_label]

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        # for idx, _ in enumerate(tqdm(range(len(self.df_data)), desc='Loading Data')):
        for idx in tqdm((self.df_data.index), desc='Loading Data'):
            path_preproc_idx = self.df_data.columns.get_loc("path_preproc_new")
            img_path = self.img_data_dir + self.df_data.iloc[idx, path_preproc_idx]
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.df_data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        # image = torch.from_numpy(sample['image'])
        image = T.ToTensor()(sample['image'])
        label = torch.from_numpy(sample['label'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        # image = imread(sample['image_path']).astype(np.float32)
        try:
            image = Image.open(sample['image_path']).convert('RGB') #PIL image
        except:
            print('PIL not working on image: {}'.format(sample['image_path']))
            image = imread(sample['image_path']).astype(np.float32)


        return {'image': image, 'label': sample['label']}

    def exam_augmentation(self,item):
        assert self.do_augment == True, 'No need for non-augmentation experiments'

        sample = self.get_sample(item) #PIL
        image = T.ToTensor()(sample['image'])

        if self.do_augment:
            image_aug = self.augment(image)

        image_all = torch.cat((image,image_aug),axis= 1)
        assert image_all.shape[1]==self.image_size*2, 'image_all.shape[1] = {}'.format(image_all.shape[1])
        return image_all



class CheXpertDataResampleModule(pl.LightningDataModule):
    def __init__(self, img_data_dir, 
                 csv_file_img, 
                 image_size, 
                 pseudo_rgb, 
                 batch_size, 
                 num_workers, 
                 augmentation,
                 outdir, 
                 version_no, 
                 chose_disease='Pleural effusion', 
                 random_state=None, 
                 num_classes=None,
                 num_per_patient=None, 
                 prevalence_setting='separate', 
                 isFlip=False):
        
        super().__init__()
        self.disease_labels_list = DISEASE_LABELS_CHE
        self.img_data_dir = img_data_dir
        self.csv_file_img = csv_file_img
        self.isFlip = isFlip

        self.outdir = outdir
        self.version_no = version_no

        # Parameters
        self.num_per_patient = num_per_patient
        if self.num_per_patient is not None:
            assert self.num_per_patient >= 1
        
        self.col_name_patient_id = 'patient_id'
        self.prevalence_setting = prevalence_setting
        assert self.prevalence_setting in ['separate', 'total', 'equal']

        self.chose_disease = chose_disease
        self.rs = random_state

        # Predefined splits
        self.perc_train, self.perc_val, self.perc_test = 0.6, 0.1, 0.3
        assert self.perc_val + self.perc_test + self.perc_train == 1

        self.disease_prevalence_total_pw = self.get_prevalence_patientwise()

        print(f"self.outdir: {self.outdir}", flush=True)
        print(f"Files in {self.outdir}: {os.listdir(self.outdir)}", flush=True)
        print(f"self.isFlip: {self.isFlip}", flush=True)

        if 'train_flip.version_0.csv' not in os.listdir(self.outdir) and self.isFlip:
            raise Exception('If doing label flipping experiments, you should have the csv files ready')

        if 'train.version_0.csv' not in os.listdir(self.outdir): 
            print('-'*30)
            print('Start sampling... Will take a while')
            df_train, df_valid, df_test = self.dataset_sampling()
        else:
            print('-'*30)
            print(f'No need to sampling, get sampling from {self.outdir}')
            df_train, df_valid, df_test = self.get_sampling(self.outdir)

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        if self.df_train is None: 
            return  # Random state does not provide enough samples to sample

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation

        if num_classes == 1:
            single_label = self.chose_disease
        else:
            single_label = None

        self.train_set = ChexpertDatasetNew(self.img_data_dir, self.df_train, self.image_size, augmentation=augmentation,
                                            pseudo_rgb=pseudo_rgb, single_label=single_label,
                                            disease_labels_list=self.disease_labels_list)
        self.val_set = ChexpertDatasetNew(self.img_data_dir, self.df_valid, self.image_size, augmentation=False,
                                          pseudo_rgb=pseudo_rgb, single_label=single_label,
                                          disease_labels_list=self.disease_labels_list)
        self.test_set = ChexpertDatasetNew(self.img_data_dir, self.df_test, self.image_size, augmentation=False,
                                           pseudo_rgb=pseudo_rgb, single_label=single_label,
                                           disease_labels_list=self.disease_labels_list)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_dataloader_nonshuffle(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_sampling(self, from_dir):
        version_number = 0 
        if not self.isFlip:
            train_set = pd.read_csv(os.path.join(from_dir, f'train.version_{version_number}.csv'))
            val_set = pd.read_csv(os.path.join(from_dir, f'val.version_{version_number}.csv'))
            test_set = pd.read_csv(os.path.join(from_dir, f'test.version_{version_number}.csv'))
        else:
            train_set = pd.read_csv(os.path.join(from_dir, f'train_flip.version_{version_number}.csv'))
            val_set = pd.read_csv(os.path.join(from_dir, f'val_flip.version_{version_number}.csv'))
            test_set = pd.read_csv(os.path.join(from_dir, f'test_flip.version_{version_number}.csv'))
        return train_set, val_set, test_set

    def dataset_sampling(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        for each_l in self.disease_labels_list:
            df[each_l] = df[each_l].apply(lambda x: 1 if x == 1 else 0)

        patient_id_list = list(set(df[self.col_name_patient_id].to_list()))
        patient_id_list.sort()

        print('#' * 30)
        print(patient_id_list[:10])
        print('#' * 30)
        sampled_df = None
        patient_info_column_names = ['pid', 'averaged_disease_label']
        patient_info_df = pd.DataFrame(columns=patient_info_column_names)

        for each_pid in tqdm(patient_id_list):
            df_this_pid = df[df[self.col_name_patient_id] == each_pid]
            len_this_pid = len(df_this_pid)

            N = len_this_pid

            sampled_this_pid = self.prioritize_sampling(df_this_pid, N=N)
            if sampled_df is None:
                sampled_df = sampled_this_pid
            else:
                sampled_df = pd.concat([sampled_df, sampled_this_pid], axis=0)
            assert len(sampled_this_pid.columns) == len(sampled_df.columns)

            averaged_disease_label = sampled_this_pid[self.chose_disease].mean()
            data = [[each_pid, averaged_disease_label]]
            df_tmp = pd.DataFrame(data=data, columns=patient_info_column_names)
            patient_info_df = pd.concat([patient_info_df, df_tmp])

        patient_info_df.reset_index(inplace=True)
        sampled_df.reset_index(inplace=True)

        print('#' * 30)
        print(f'sampled_df: {len(sampled_df)}')
        print(sampled_df)
        print('#' * 30)
        print(f'patient_info_df: {len(patient_info_df)}')
        print(patient_info_df)
        print('#' * 30)

        train_set, val_set, test_set = self.set_split(patient_info_df, self.perc_train, self.perc_val, self.perc_test, self.rs)

        train_set.reset_index(inplace=True, drop=True)
        val_set.reset_index(inplace=True, drop=True)
        test_set.reset_index(inplace=True, drop=True)

        train_set.to_csv(os.path.join(self.outdir, f'train.version_{self.version_no}.csv'), index=False)
        val_set.to_csv(os.path.join(self.outdir, f'val.version_{self.version_no}.csv'), index=False)
        test_set.to_csv(os.path.join(self.outdir, f'test.version_{self.version_no}.csv'), index=False)

        return train_set, val_set, test_set

    def get_prevalence(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        for each_l in self.disease_labels_list:
            df[each_l] = df[each_l].apply(lambda x: 1 if x == 1 else 0)

        df_per_patient = df.groupby([self.col_name_patient_id])[self.disease_labels_list].mean()
        print('DEBUG', df_per_patient)

        df_per_patient_p = df_per_patient.mean().to_list()

        dict_per_patient_p = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_p[each_l] = df_per_patient_p[i]

        print('Disease prevalence total: {}'.format(dict_per_patient_p))

        return dict_per_patient_p

    def get_prevalence_patientwise(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        for each_l in self.disease_labels_list:
            df[each_l] = df[each_l].apply(lambda x: 1 if x == 1 else 0)

        df_numeric = df.select_dtypes(include=[float, int])
        df_per_patient = df.groupby([self.col_name_patient_id])[df_numeric.columns].mean()
        
        for each_labels in self.disease_labels_list:
            df_per_patient[each_labels] = df_per_patient[each_labels].apply(lambda x: 1 if x > 0 else 0)

        df_per_patient_p = df_per_patient.mean()[self.disease_labels_list].to_list()

        dict_per_patient_p = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_p[each_l] = df_per_patient_p[i]

        print('PATIENT WISE disease prevalence')
        print('Disease prevalence total: {}'.format(dict_per_patient_p))

        return dict_per_patient_p

    def prioritize_sampling(self, df, N):
        return df

    def set_split(self, df, train_frac, val_frac, test_frac, rs):
        test = df.sample(frac=test_frac, random_state=rs)
        train_val = df.drop(index=test.index)
        train = train_val.sample(frac=train_frac / (train_frac + val_frac), random_state=rs)
        val = train_val.drop(index=train.index)
        return train, val, test
