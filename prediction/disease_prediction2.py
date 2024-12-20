import sys
sys.path.append('../../detecting_causes_of_gender_bias_chest_xrays')

from prediction.models import ResNet,DenseNet
from dataloader.dataloader2 import DISEASE_LABELS_CHE,CheXpertDataResampleModule
from dataloader.dataloader import NIHDataResampleModule, DISEASE_LABELS_NIH

import os
import torch
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
import shutil
import numpy as np


hp_default_value={'model':'resnet',
                  'model_scale':'18',
                  'lr':1e-6,
                  'bs':64,
                  'epochs':50,
                  'pretrained':True,
                  'augmentation':True,
                  'is_multilabel':False,
                  'image_size':(224,224),
                  'crop':None,
                  'prevalence_setting':'separate',
                  'save_model':False,
                  'num_workers':2,
                  'num_classes':1

}


def get_cur_version(dir_path):
    i = 0
    while os.path.exists(dir_path + '/version_{}'.format(i)):
        i += 1
    return i


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test_func(args, model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0, args.num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()


def main(args, random_state=None, chose_disease_str=None):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.dev) if use_cuda else "cpu")
    print('DEVICE:{}'.format(device))

    # get run_config
    run_config = f'{args.dataset}-{chose_disease_str}'  # dataset and the predicted label
    run_config += f'-npp{args.npp}-rs{random_state}'  # npp and rs

    # if the hp value is not default
    args_dict = vars(args)
    for each_hp in hp_default_value.keys():
        if hp_default_value[each_hp] != args_dict[each_hp]:
            run_config += f'-{each_hp}{args_dict[each_hp]}'

    print('------------------------------------------\n' * 3)
    print('run_config:{}'.format(run_config))

    # Create output directory
    run_dir = '/work3/s206182/run/chexpert/'
    out_dir = run_dir + run_config
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cur_version = get_cur_version(out_dir)

    if args.dataset == 'NIH':
        data = NIHDataResampleModule(img_data_dir=args.img_data_dir,
                                     csv_file_img=args.csv_file_img,
                                     image_size=args.image_size,
                                     pseudo_rgb=False,
                                     batch_size=args.bs,
                                     num_workers=args.num_workers,
                                     augmentation=args.augmentation,
                                     outdir=out_dir,
                                     version_no=cur_version,
                                     chose_disease=chose_disease_str,
                                     random_state=random_state,
                                     num_classes=args.num_classes,
                                     num_per_patient=args.npp,
                                     crop=args.crop,
                                     prevalence_setting=args.prevalence_setting,
                                     )
    elif args.dataset == 'chexpert':
        if args.crop is not None:
            raise Exception('Crop experiment not implemented for chexpert.')
        data = CheXpertDataResampleModule(img_data_dir=args.img_data_dir,
                                          csv_file_img=args.csv_file_img,
                                          image_size=args.image_size,
                                          pseudo_rgb=False,
                                          batch_size=args.bs,
                                          num_workers=args.num_workers,
                                          augmentation=args.augmentation,
                                          outdir=out_dir,
                                          version_no=cur_version,
                                          chose_disease=chose_disease_str,
                                          random_state=random_state,
                                          num_classes=args.num_classes,
                                          num_per_patient=args.npp,
                                          prevalence_setting=args.prevalence_setting,
                                          isFlip=args.flip,
                                          )
    else:
        raise Exception('not implemented')

    # model
    if args.model == 'resnet':
        model_type = ResNet
    elif args.model == 'densenet':
        model_type = DenseNet
    model = model_type(num_classes=args.num_classes, lr=args.lr, pretrained=args.pretrained, model_scale=args.model_scale,
                       loss_func_type='BCE')

    temp_dir = os.path.join(out_dir, 'temp_version_{}'.format(cur_version))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0, 5):
        if args.augmentation:
            sample = data.train_set.exam_augmentation(idx)
            sample = np.asarray(sample)
            sample = np.transpose(sample, (2, 1, 0))
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.png'), sample)
        else:
            sample = data.train_set.get_sample(idx)  # PIL
            sample = np.asarray(sample['image'])
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.png'), sample.astype(np.uint8))

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        log_every_n_steps=1,
        max_epochs=args.epochs,
        gpus=args.gpus,
        accelerator="auto",
        logger=TensorBoardLogger(run_dir, name=run_config, version=cur_version),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0, args.num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, args.num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, args.num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test_func(args, model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, f'predictions.val.version_{cur_version}.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test_func(args, model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, f'predictions.test.version_{cur_version}.csv'), index=False)

    print('TESTING on train set')
    preds_test, targets_test, logits_test = test_func(args, model, data.train_dataloader_nonshuffle(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, f'predictions.train.version_{cur_version}.csv'), index=False)

    if not args.save_model:
        model_para_dir = os.path.join(out_dir, 'version_{}'.format(cur_version))
        shutil.rmtree(model_para_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    parser.add_argument('-s', '--dataset', default='NIH', help='Dataset', choices=['NIH', 'chexpert'])
    parser.add_argument('-d', '--disease_label', default=['Pneumothorax'], help='Chosen disease label', type=str, nargs='*')
    parser.add_argument('-n', '--npp', default=1, help='Number per patient, could be integer or None (no sampling)', type=lambda x: None if x.lower() == 'none' else int(x))
    parser.add_argument('-r', '--random_state', default='0-10', help='random state')
    parser.add_argument('-p', '--img_dir', help='your img dir path here', type=str)
    parser.add_argument('--flip', default=False, help='whether using flip labels', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--lr', default=1e-6, help='learning rate, default=1e-6')
    parser.add_argument('--bs', default=64, help='batch size, default=64')
    parser.add_argument('--epochs', default=50, help='number of epochs, default=50')
    parser.add_argument('--model', default='resnet', help='model, default=\'ResNet\'')
    parser.add_argument('--model_scale', default='18', help='model scale, default=18', type=str)
    parser.add_argument('--pretrained', default=True, help='pretrained or not, True or False, default=True', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--augmentation', default=True, help='augmentation during training or not, True or False, default=True', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--is_multilabel', default=False, help='training with multilabel or not, default=False', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--image_size', default=224, help='image size', type=int)
    parser.add_argument('--crop', default=None, help='crop the bottom part of the image')
    parser.add_argument('--prevalence_setting', default='separate', help='choose from [separate, equal, total]', choices=['separate', 'equal', 'total'])
    parser.add_argument('--save_model', default=False, help='save model parameter or not', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--num_workers', default=2, help='number of workers')

    args = parser.parse_args()

    if args.is_multilabel:
        args.num_classes = len(DISEASE_LABELS_NIH) if args.dataset == 'NIH' else len(DISEASE_LABELS_CHE)
    else:
        args.num_classes = 1

    if args.image_size == 224:
        args.img_data_dir = args.img_dir+'{}/preproc_224x224/'.format(args.dataset)
    elif args.image_size == 1024:
        args.img_data_dir = args.img_dir+'{}/images/'.format(args.dataset)

    if args.dataset == 'NIH':
        args.csv_file_img = '../datafiles/Data_Entry_2017_v2020_clean_split.csv'
    elif args.dataset == 'chexpert':
        args.csv_file_img = '../datafiles/chexpert.sample.allrace.csv'
    else:
        raise Exception('Not implemented.')

    print('hyper-parameters:')
    print(args)

    args.epochs = int(args.epochs)
    if len(args.random_state.split('-')) == 2:
        rs_min, rs_max = map(int, args.random_state.split('-'))
    else:
        rs_min, rs_max = int(args.random_state), int(args.random_state) + 1

    disease_label_list = args.disease_label
    if len(disease_label_list) == 1 and disease_label_list[0] == 'all':
        disease_label_list = DISEASE_LABELS_NIH if args.dataset == 'NIH' else DISEASE_LABELS_CHE
    print('disease_label_list:{}'.format(disease_label_list))

    print('***********RESAMPLING EXPERIMENT**********\n')
    for d in disease_label_list:
        for i in np.arange(rs_min, rs_max):
            main(args, random_state=i, chose_disease_str=d)
