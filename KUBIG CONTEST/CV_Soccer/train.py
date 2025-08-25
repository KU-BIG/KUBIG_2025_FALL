import importlib
import os
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#python train.py --formatDataset --datasetType successivenobg --training D:\\DeepSport\\CodeForGithub\\FullDataset\\listgame_Train_300.npy --validation D:\\DeepSport\\CodeForGithub\\FullDataset\\listgame_Valid_100.npy --testing D:\\DeepSport\\CodeForGithub\\FullDataset\\listgame_Test_100.npy --PCA --model VideoOnly2FC pooling

from dataset import Dataset
from trainer import Trainer

def main(args):
    dataset = Dataset(args.windowSize, args.featurePerSec, args.batchSize, system=args.system)

    if args.formatDataset:
        dataset.prepareDataset(args.training, args.validation, args.testing, args.featuresVideo, args.featuresAudio, args.PCA)
    else:
        dataset.loadDataset(args.training, args.validation, args.testing, args.featuresVideo, args.featuresAudio, args.PCA)

    module = importlib.import_module('networks')
    class_ = getattr(module, args.model)
    network = class_(dataset, args.pooling, VLAD_K=args.vladK)

    trainer = Trainer(dataset, network, output_prefix=args.outputModelName)
    trainer.train(args.maxEpoch, args.LR, args.tensorboarddir)

if __name__ == "__main__":
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', required=True, type=str, help="the name of the model to use")

    parser.add_argument('--training', required=True, type=str, help='the file containing the training data')
    parser.add_argument('--validation', required=True, type=str, help='the file containing the validation data')
    parser.add_argument('--testing', required=True, type=str, help='the file containing the testing data')
    parser.add_argument('--formatDataset', required=False, action='store_true', help='format and prepare the dataset')

    parser.add_argument('--featuresVideo', required=False, type=str, default='ResNET', help="the type of video features to use")
    parser.add_argument('--featuresAudio', required=False, type=str, default='VGGish', help="the type of audio features to use")
    parser.add_argument('--PCA', required=False, action='store_true', help='use PCA version of the video features')
    parser.add_argument('--featurePerSec', required=False, type=int, default=2, help="The framerate to used")

    parser.add_argument('--LR', required=False, type=float, default=0.01, help='Starting learning rate')
    parser.add_argument('--windowSize', required=False, type=int, default=60, help='The window size to use')
    parser.add_argument('--batchSize', required=False, type=int, default=60, help="Size of the batch")
    parser.add_argument('--maxEpoch', required=False, type=int, default=200, help="Maximum number of epochs")

    parser.add_argument('--tensorboarddir', required=False, type=str, default='Model', help='folder for TensorBoard logs')
    parser.add_argument('--outputModelName', required=False, type=str, default='', help='prefix of the name file for saving trained model')

    parser.add_argument('--vladK', required=False, type=int, default=64, help="Number of clusters")
    parser.add_argument('--pooling', required=False, choices=['RVLAD', 'VLAD'], type=str, default='RVLAD', help="The pooling layer")

    parser.add_argument('--seed', required=False, type=int, default=1234, help='Random seed')

    parser.add_argument('--system', required=False, type=str, default="linux", help='linux or windows')

    args = parser.parse_args()

    print(args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    start = time.time()
    main(args)
    print("Total Execution Time is {} seconds".format(time.time()-start))
