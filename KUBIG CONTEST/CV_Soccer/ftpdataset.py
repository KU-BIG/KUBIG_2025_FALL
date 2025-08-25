import json
import numpy as np
import os
import random

from pprint import pprint

import time

from ftplib import FTP
FTP_SERVER_IP = ''
FTP_USER = ''
FTP_PWD = ''
FTP_ROOT_DIR = '../data/'

def create_directory(dir_name):
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
            os.mkdir(dir_name)

class Dataset():

    def __init__(self, batch_size=60, system='linux'):
        print("Init dataset")

        self.num_classes = 4
        self.count_labels = np.array([0, 0, 0, 0])
        self.size_batch = batch_size
        self.nb_batch_training = 0
        self.nb_epoch_per_batch = 1
        self.system = system

    def prepareTrainingDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        number_frames_in_window = window_size_sec * feature_per_second
        self.number_frames_in_window = number_frames_in_window

        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        batch_dir = "Windows_size_{}_sec".format(window_size_sec)
        create_directory(batch_dir)

        print("Prepare Training dataset")
        self.training_GamesKeys=[]
        self.training_features = {}
        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}
            cpt = 0
            for featureFileName in os.listdir(gamePath):
                found = False
                if featureVideoName in featureFileName and "float16" not in featureFileName and ((PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName)):
                    featType = "video"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                elif featureAudioName in featureFileName and "float16" not in featureFileName:
                    featType = "audio"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                if found:
                    featureFullPath[featType][half_key] = os.path.join(gamePath, featureFileName)
                    cpt += 1
                    if cpt == 4:
                        break

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                self.training_GamesKeys.append(key)
                self.training_video_features_cont = np.load(featureFullPath['video'][i])
                self.training_audio_features_cont = np.load(featureFullPath['audio'][i])

                cnt_data_augmentation = 0

                l = self.training_video_features_cont.shape[0] - self.training_video_features_cont.shape[0] % number_frames_in_window

                training_features_video = np.zeros((cnt_data_augmentation + int(l/number_frames_in_window), number_frames_in_window, 512))
                training_features_audio = np.zeros((cnt_data_augmentation + int(l/number_frames_in_window), number_frames_in_window, 512))

                cnt_data_augmentation = 0
                for minframe in np.reshape(self.training_video_features_cont[0:l,:], (-1, number_frames_in_window, 512)):
                    training_features_video[cnt_data_augmentation] = minframe
                    cnt_data_augmentation += 1

                cnt_data_augmentation = 0
                for minframe in np.reshape(self.training_audio_features_cont[0:l,:], (-1, number_frames_in_window, 512)):
                    training_features_audio[cnt_data_augmentation] = minframe
                    cnt_data_augmentation += 1

                labelFullPath = os.path.join(gamePath, "Labels.json")
                with open(labelFullPath) as labelFile:
                    jsonLabel = json.loads(labelFile.read())

                Labels = np.zeros((training_features_video.shape[0], 4), dtype=int)
                Labels[:,0] = 1

                for event in jsonLabel['annotations']:
                    Time_Half = int(event["gameTime"][0])
                    Time_Minute = int(event["gameTime"][-5:-3])
                    Time_Second = int(event["gameTime"][-2:])

                    if ("card" in event["label"]): label = 1
                    elif ("subs" in event["label"]): label = 2
                    elif ("soccer" in event["label"]): label = 3
                    else: label = 0

                    if Time_Half == i+1:
                        index = min(
                            int(60/window_size_sec * Time_Minute + int(Time_Second / window_size_sec)),
                            Labels.shape[0]-1
                        )

                        Labels[index, 0] = 0
                        Labels[index, label] = 1

                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'

                video_filename = os.path.join(batch_dir, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                np.save(video_filename, training_features_video, allow_pickle=True)
                audio_filename = os.path.join(batch_dir, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                np.save(audio_filename, training_features_audio, allow_pickle=True)
                label_filename = os.path.join(batch_dir, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))
                np.save(label_filename, Labels, allow_pickle=True)

                self.training_features[key] = dict()
                self.training_features[key]['video'] = video_filename
                self.training_features[key]['audio'] = audio_filename
                self.training_features[key]['label'] = label_filename

    def loadTrainingDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        number_frames_in_window = window_size_sec * feature_per_second
        self.number_frames_in_window = number_frames_in_window

        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        batch_dir = "Windows_size_{}_sec".format(window_size_sec)
        create_directory(batch_dir)

        print("Load Training dataset")
        self.training_GamesKeys=[]
        self.training_features = {}
        for gamePath in listGames: # gamePath : england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                self.training_GamesKeys.append(key)

                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'
                self.training_features[key] = dict()
                video_filename = os.path.join(batch_dir, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                self.training_features[key]['video'] = video_filename
                audio_filename = os.path.join(batch_dir, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                self.training_features[key]['audio'] = audio_filename
                label_filename = os.path.join(batch_dir, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))
                self.training_features[key]['label'] = label_filename

    def prepareValidationDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        number_frames_in_window = window_size_sec * feature_per_second

        print("Prepare Validation dataset")
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        batch_dir = "Windows_size_{}_sec".format(window_size_sec)

        self.validation_GamesKeys=[]
        self.validation_features = {}

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}
            cpt = 0
            for featureFileName in os.listdir(gamePath):
                found = False
                if featureVideoName in featureFileName and "float16" not in featureFileName and ((PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName)):
                    featType = "video"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                elif featureAudioName in featureFileName and "float16" not in featureFileName:
                    featType = "audio"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                if found:
                    featureFullPath[featType][half_key] = os.path.join(gamePath, featureFileName)
                    cpt += 1
                    if cpt == 4:
                        break

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                self.validation_GamesKeys.append(key)
                validation_features_video = np.load(featureFullPath['video'][i])
                validation_features_audio = np.load(featureFullPath['audio'][i])

                l = validation_features_video.shape[0] - validation_features_video.shape[0] % number_frames_in_window
                validation_features_video = np.reshape(validation_features_video[0:l,:], (-1, number_frames_in_window, 512))
                validation_features_audio = np.reshape(validation_features_audio[0:l,:], (-1, number_frames_in_window, 512))

                labelFullPath = os.path.join(gamePath, "Labels.json")
                with open(labelFullPath) as labelFile:
                    jsonLabel = json.loads(labelFile.read())

                Labels = np.zeros((validation_features_video.shape[0], 4), dtype=int)
                Labels[:,0] = 1

                for event in jsonLabel['annotations']:
                    Time_Half = int(event["gameTime"][0])
                    Time_Minute = int(event["gameTime"][-5:-3])
                    Time_Second = int(event["gameTime"][-2:])

                    if ("card" in event["label"]): label = 1
                    elif ("subs" in event["label"]): label = 2
                    elif ("soccer" in event["label"]): label = 3
                    else: label = 0

                    if Time_Half == i+1:
                        index = min(
                            int(60/window_size_sec * Time_Minute + int(Time_Second / window_size_sec)),
                            Labels.shape[0]-1
                        )

                        Labels[index, 0] = 0
                        Labels[index, label] = 1

                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'
                video_filename = os.path.join(batch_dir, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                np.save(video_filename, validation_features_video, allow_pickle=True)
                audio_filename = os.path.join(batch_dir, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                np.save(audio_filename, validation_features_audio, allow_pickle=True)
                label_filename = os.path.join(batch_dir, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))
                np.save(label_filename, Labels, allow_pickle=True)

                self.validation_features[key] = dict()
                self.validation_features[key]['video'] = video_filename
                self.validation_features[key]['audio'] = audio_filename
                self.validation_features[key]['label'] = label_filename
        self.nb_batch_validation = len(self.validation_GamesKeys)

    def loadValidationDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        number_frames_in_window = window_size_sec * feature_per_second

        print("Load Validation dataset")
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        batch_dir = "Windows_size_{}_sec".format(window_size_sec)

        self.validation_GamesKeys=[]
        self.validation_features = {}

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                self.validation_GamesKeys.append(key)

                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'
                self.validation_features[key] = dict()
                video_filename = os.path.join(batch_dir, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                self.validation_features[key]['video'] = video_filename
                audio_filename = os.path.join(batch_dir, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                self.validation_features[key]['audio'] = audio_filename
                label_filename = os.path.join(batch_dir, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))
                self.validation_features[key]['label'] = label_filename
        self.nb_batch_validation = len(self.validation_GamesKeys)

    def prepareTestingDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        self.number_frames_in_window = window_size_sec * feature_per_second
        self.featureVideoName = featureVideoName
        self.featureAudioName = featureAudioName

        self.PCA = PCA

        print("Prepare Testing dataset")
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        batch_dir = "Windows_size_{}_sec".format(window_size_sec)

        self.testing_GamesKeys = []
        self.testing_features = {}

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}
            cpt = 0
            for featureFileName in os.listdir(gamePath):
                found = False
                if featureVideoName in featureFileName and "float16" not in featureFileName and ((PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName)):
                    featType = "video"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                elif featureAudioName in featureFileName and "float16" not in featureFileName:
                    featType = "audio"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                if found:
                    featureFullPath[featType][half_key] = os.path.join(gamePath, featureFileName)
                    cpt += 1
                    if cpt == 4:
                        break

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                self.testing_GamesKeys.append(key)
                testing_features_video = np.load(featureFullPath['video'][i])
                testing_features_audio = np.load(featureFullPath['audio'][i])

                l = testing_features_video.shape[0] - testing_features_video.shape[0] % self.number_frames_in_window
                testing_features_video = np.reshape(testing_features_video[0:l,:], (-1, self.number_frames_in_window, 512))
                testing_features_audio = np.reshape(testing_features_audio[0:l,:], (-1, self.number_frames_in_window, 512))

                labelFullPath = os.path.join(gamePath, "Labels.json")
                with open(labelFullPath) as labelFile:
                    jsonLabel = json.loads(labelFile.read())

                Labels = np.zeros((testing_features_video.shape[0], 4), dtype=int)
                Labels[:,0] = 1

                for event in jsonLabel['annotations']:
                    Time_Half = int(event["gameTime"][0])
                    Time_Minute = int(event["gameTime"][-5:-3])
                    Time_Second = int(event["gameTime"][-2:])

                    if ("card" in event["label"]): label = 1
                    elif ("subs" in event["label"]): label = 2
                    elif ("soccer" in event["label"]): label = 3
                    else: label = 0

                    if Time_Half == i+1:
                        index = min(
                            int(60/window_size_sec * Time_Minute + int(Time_Second / window_size_sec)),
                            Labels.shape[0]-1
                        )

                        Labels[index, 0] = 0
                        Labels[index, label] = 1

                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'
                video_filename = os.path.join(batch_dir, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                np.save(video_filename, testing_features_video, allow_pickle=True)
                audio_filename = os.path.join(batch_dir, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                np.save(audio_filename, testing_features_audio, allow_pickle=True)
                label_filename = os.path.join(batch_dir, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))
                np.save(label_filename, Labels, allow_pickle=True)

                self.testing_features[key] = dict()
                self.testing_features[key]['video'] = video_filename
                self.testing_features[key]['audio'] = audio_filename
                self.testing_features[key]['label'] = label_filename
        self.weights = [1, 1, 1, 1]

        self.nb_batch_testing = len(self.testing_GamesKeys)

    def loadTestingDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        self.number_frames_in_window = window_size_sec * feature_per_second
        self.featureVideoName = featureVideoName
        self.featureAudioName = featureAudioName

        self.PCA = PCA

        print("Load Testing dataset")
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        batch_dir = "Windows_size_{}_sec".format(window_size_sec)

        self.testing_GamesKeys = []
        self.testing_features = {}

        for gamePath in listGames:
            gamePath = os.path.join(path_data_dirname, gamePath)

            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}

            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                if self.system == "windows":
                    key = key.replace('/', '\\')
                self.testing_GamesKeys.append(key)

                if self.system == "linux":
                    sep = '/'
                else:
                    sep = '\\'
                self.testing_features[key] = dict()
                video_filename = os.path.join(batch_dir, "{}_video_{}.npy".format('_'.join(key.split(sep)[-4:]), featureVideoName))
                self.testing_features[key]['video'] = video_filename
                audio_filename = os.path.join(batch_dir, "{}_audio_{}.npy".format('_'.join(key.split(sep)[-4:]), featureAudioName))
                self.testing_features[key]['audio'] = audio_filename
                label_filename = os.path.join(batch_dir, "{}_labels.npy".format('_'.join(key.split(sep)[-4:])))
                self.testing_features[key]['label'] = label_filename

        self.weights = [1, 1, 1, 1]
        self.nb_batch_testing = len(self.testing_GamesKeys)

    def prepareNewEpoch(self):
        random.shuffle(self.training_GamesKeys)

        nb_halves = len(self.training_GamesKeys)
        print("size_batch:", self.size_batch)
        print("nb_halves:", nb_halves)

        self.nb_batch_training = int(np.ceil(nb_halves/self.size_batch))

        self._current_training_batch_index = -1
        self._current_validation_batch_index = -1

        self.counting = [0,0,0,0]

    def getTrainingBatch(self, num_batch):
        self._current_training_batch_index = num_batch

        init_games = num_batch * self.size_batch
        end_games = min((num_batch+1)*self.size_batch, len(self.training_GamesKeys))

        return self.getGamesBatch(init_games, end_games)

    def getGamesBatch(self, init_games, end_games):
        key = self.training_GamesKeys[init_games]
        train_batch_video_features = np.load(self.training_features[key]['video'])
        train_batch_audio_features = np.load(self.training_features[key]['audio'])
        train_batch_labels = np.load(self.training_features[key]['label'])

        train_batch_indices = []

        print("from", init_games, "to", end_games)

        for gameKey in self.training_GamesKeys[init_games+1:end_games]:
            train_batch_video_features = np.concatenate((train_batch_video_features, np.load(self.training_features[gameKey]['video'])))
            train_batch_audio_features = np.concatenate((train_batch_audio_features, np.load(self.training_features[gameKey]['audio'])))
            train_batch_labels = np.concatenate((train_batch_labels, np.load(self.training_features[gameKey]['label'])))

        return train_batch_video_features, train_batch_audio_features, train_batch_labels, train_batch_indices

    def getValidationBatch(self, num_batch):
        valid_batch_features = self.validation_features[self.validation_GamesKeys[num_batch]]
        valid_batch_video_features = np.load(valid_batch_features['video'])
        valid_batch_audio_features = np.load(valid_batch_features['audio'])
        valid_batch_labels = np.load(valid_batch_features['label'])

        self._current_validation_batch_index = num_batch

        return valid_batch_video_features, valid_batch_audio_features, valid_batch_labels

    def getTestingBatch(self, num_batch):
        testing_batch_features = self.testing_features[self.testing_GamesKeys[num_batch]]
        testing_batch_video_features = np.load(testing_batch_features['video'])
        testing_batch_audio_features = np.load(testing_batch_features['audio'])
        testing_batch_labels = np.load(testing_batch_features['label'])

        self._current_testing_batch_index = num_batch

        return testing_batch_video_features, testing_batch_audio_features, testing_batch_labels

    def loadSpottingTestingDataset(self, path_data, featureVideoName, featureAudioName, PCA=True, window_size_sec=60, feature_per_second=2):
        self.window_size_sec = window_size_sec
        self.featurePerSecond = feature_per_second
        self.number_frames_in_window = window_size_sec * feature_per_second
        self.featureVideoName = featureVideoName
        self.featureAudioName = featureAudioName

        self.PCA = PCA

        ftp = FTP(FTP_SERVER_IP)
        ftp.login(user=FTP_USER, passwd=FTP_PWD)
        ftp.cwd(FTP_ROOT_DIR)

        print("Prepare Action Spotting Testing dataset")
        path_data_dirname, path_data_basename = os.path.split(path_data)

        listGames = np.load(path_data)

        self.testing_GamesKeys = []
        self.testing_features = {}

        for gamePath in listGames:
            featureFullPath = {"video": ["", ""], "audio": ["", ""], "keys": ["Half_1", "Half_2"]}
            cpt = 0
            ftp.cwd(FTP_ROOT_DIR)
            ftp.cwd(gamePath)
            for featureFileName in ftp.nlst():
                found = False
                if featureVideoName in featureFileName and "float16" not in featureFileName and ((PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName)):
                    featType = "video"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                elif featureAudioName in featureFileName and "float16" not in featureFileName:
                    featType = "audio"
                    if "1_" in featureFileName:
                        half_key = 0
                        found = True
                    elif "2_" in featureFileName:
                        half_key = 1
                        found = True
                if found:
                    key_path = '_'.join([gamePath.replace('/','_'), featureFileName])
                    ftp.retrbinary("RETR " + featureFileName, open(key_path, 'wb').write)
                    featureFullPath[featType][half_key] = key_path
                    cpt += 1
                    if cpt == 4:
                        break

            key_path = '_'.join([gamePath.replace('/','_'), "Labels.json"])
            ftp.retrbinary("RETR Labels.json", open(key_path, 'wb').write)
            for i in range(2):
                key = os.path.join(gamePath, featureFullPath['keys'][i])
                self.testing_GamesKeys.append(key)
                self.testing_features[key] = {}
                self.testing_features[key]['video'] = featureFullPath['video'][i]
                self.testing_features[key]['audio'] = featureFullPath['audio'][i]
                self.testing_features[key]['label'] = key_path
                self.testing_features[key]['half'] = i

        self.weights = [1, 1, 1, 1]

        self.nb_batch_testing = len(self.testing_GamesKeys)

    def remove_files(self):
        for key in self.testing_GamesKeys:
            os.remove(self.testing_features[key]['video'])
            os.remove(self.testing_features[key]['audio'])
            if os.path.exists(self.testing_features[key]['label']):
                os.remove(self.testing_features[key]['label'])

    def getSpottingTestingBatch(self, num_batch):
        testing_batch_features = self.testing_features[self.testing_GamesKeys[num_batch]]

        # Video
        testing_features_cont = np.load(testing_batch_features['video'])

        strides = testing_features_cont.strides
        nb_frames = testing_features_cont.shape[0]
        size_feature = testing_features_cont.shape[1]
        sliding_window_seconds = self.window_size_sec
        sliding_window_frame = sliding_window_seconds * self.featurePerSecond

        testing_features_cont = np.append([testing_features_cont[0,:]]*sliding_window_seconds, testing_features_cont, axis=0)
        testing_features_cont = np.append(testing_features_cont, [testing_features_cont[-1,:]]*sliding_window_seconds, axis=0)

        testing_batch_video_features = np.lib.stride_tricks.as_strided(testing_features_cont, shape=(int(nb_frames/2), sliding_window_frame, size_feature), strides=(strides[0]*2,strides[0],strides[1]))

        for pl in range(20, 40):
            for pk in range(10):
                assert((testing_batch_video_features[pl,pk,:] - testing_features_cont[pl*2+pk,:]).sum() == 0)

        # Audio
        testing_features_cont = np.load(testing_batch_features['audio'])

        strides = testing_features_cont.strides
        nb_frames = testing_features_cont.shape[0]
        size_feature = testing_features_cont.shape[1]
        sliding_window_seconds = self.window_size_sec
        sliding_window_frame = sliding_window_seconds * self.featurePerSecond

        testing_features_cont = np.append([testing_features_cont[0,:]]*sliding_window_seconds, testing_features_cont, axis=0)
        testing_features_cont = np.append(testing_features_cont, [testing_features_cont[-1,:]]*sliding_window_seconds, axis=0)

        testing_batch_audio_features = np.lib.stride_tricks.as_strided(testing_features_cont, shape=(int(nb_frames/2), sliding_window_frame, size_feature), strides=(strides[0]*2,strides[0],strides[1]))

        for pl in range(20, 40):
            for pk in range(10):
                assert((testing_batch_audio_features[pl,pk,:] - testing_features_cont[pl*2+pk,:]).sum() == 0)

        # Labels
        labelFullPath = testing_batch_features['label']
        with open(labelFullPath) as labelFile:
            jsonLabel = json.loads(labelFile.read())

        Labels = np.zeros((testing_batch_audio_features.shape[0], 4), dtype=int)
        Labels[:,0] = 1

        for event in jsonLabel["annotations"]:
            Half = int(event['gameTime'][0])
            Time_Minute = int(event['gameTime'][-5:-3])
            Time_Second = int(event['gameTime'][-2:])

            if ("card" in event['label']): label = 1
            elif ("subs" in event['label']): label = 2
            elif ("soccer" in event['label']): label = 3
            else: label = 0

            if Half == testing_batch_features['half']+1:
                aroundValue = min(
                    Time_Minute * 60 + Time_Second,
                    Labels.shape[0] - 1
                )
                Labels[(aroundValue - int(sliding_window_seconds/2)):(aroundValue + int(sliding_window_seconds/2)), 0] = 0
                Labels[(aroundValue - int(sliding_window_seconds/2)):(aroundValue + int(sliding_window_seconds/2)), label] = 1

        testing_batch_labels = Labels

        self._current_testing_batch_index = num_batch

        return testing_batch_video_features, testing_batch_audio_features, testing_batch_labels, self.testing_GamesKeys[num_batch]
