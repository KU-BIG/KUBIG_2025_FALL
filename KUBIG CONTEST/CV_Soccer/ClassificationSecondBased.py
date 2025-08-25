import importlib
import logging
import numpy as np
import os
import torch
import time
from tqdm import tqdm
from sklearn.metrics import average_precision_score, confusion_matrix

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import Dataset as Dataset
from ftpdataset import Dataset as FTPDataset
# 'networks' 모듈에 정의된 클래스들이 import된다고 가정합니다.

def main(args):
    print("Loading Testing Data:", args.testing)
    print("args.PCA:", args.PCA)
    print("args.featuresVideo:", args.featuresVideo)
    print("args.featuresAudio:", args.featuresAudio)
    print("args.network:", args.network)
    print("args.VLAD_k:", args.VLAD_k)
    print("Architecture:", args.architecture)
    print("args.system:", args.system)
    print("flush!", flush=True)

    if args.FTP:
        dataset = FTPDataset()
    else:
        dataset = Dataset(
            window_size_sec=args.WindowSize,
            # 'feature_per_second'와 'batch_size'는 코드에 명시되어 있지 않으므로
            # 적절한 기본값을 설정하거나 추가적인 인자를 받아야 합니다.
            # 예시로 2와 32를 사용하겠습니다. 실제 값에 맞게 수정해야 합니다.
            feature_per_second=2,
            batch_size=32, 
            system=args.system
        )

    dataset.loadSpottingTestingDataset(
        path_data=args.testing,
        featureVideoName=args.featuresVideo,
        featureAudioName=args.featuresAudio,
        PCA=args.PCA,
        window_size_sec=args.WindowSize
    )

    module = importlib.import_module('networks')
    class_ = getattr(module, args.architecture)
    network = class_(dataset, args.network, VLAD_K=args.VLAD_k)

    # ------------------ 모델 가중치 로드 시작 ------------------
    # 여기에 학습을 통해 저장된 모델 파일의 실제 경로를 입력하세요.
    MODEL_PATH = "Model/vlad-archi5-20sec_2025-08-18_20-07-22_epoch_2_model.pth" 
    
    try:
        # 모델의 가중치(state_dict)를 불러옵니다.
        state_dict = torch.load(MODEL_PATH,  weights_only=True)
        
        # 네트워크에 가중치를 적용합니다.
        network.load_state_dict(state_dict)
        
        print(f"✅ 사전 학습된 모델이 성공적으로 로드되었습니다: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"오류: 모델 파일이 지정된 경로에 존재하지 않습니다: {MODEL_PATH}")
    except Exception as e:
        print(f"모델 로딩 중 오류가 발생했습니다: {e}")
        
    # ------------------ 모델 가중치 로드 끝 ------------------

    # PyTorch에서는 명시적인 세션이 필요 없습니다.
    # 대신 GPU 사용 여부를 설정하고 모델을 해당 장치로 이동시킵니다.
    device = torch.device(f"cuda:{args.GPU}" if args.GPU >= 0 and torch.cuda.is_available() else "cpu")
    network.to(device)

    create_directory("data")

    # 모델 가중치 초기화 또는 로드
    # 이 부분은 networks.py 파일의 initialize 메서드에 따라 달라집니다.
    # 원본 코드의 initialize(sess)와 동일한 기능을 수행하도록 가정합니다.
    try:
        network.initialize()
    except Exception as e:
        print(f"Error during network initialization: {e}")
        # 필요에 따라 다른 초기화 로직을 추가할 수 있습니다.
        pass

    # 평가 모드로 전환 (Dropout 및 BatchNorm 비활성화)
    network.eval()
    
    # 메트릭 계산을 위한 리스트 초기화
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum') # 손실 합산을 위해 reduction='sum' 사용
    
    start_time = time.time()
    
    # gradient 계산을 비활성화하여 메모리 사용량을 줄이고 속도를 향상시킵니다.
    with torch.no_grad():
        for i in tqdm(range(dataset.nb_batch_testing)):
            try:
                batch_video_feature, batch_audio_features, batch_labels, key = dataset.getSpottingTestingBatch(i)

                # numpy 배열을 PyTorch 텐서로 변환하고 GPU로 이동
                batch_video_tensor = torch.tensor(batch_video_feature, dtype=torch.float32, device=device)
                batch_audio_tensor = torch.tensor(batch_audio_features, dtype=torch.float32, device=device)
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.float32, device=device)

                # 배치를 4개로 분할하여 처리
                split_size = batch_video_tensor.shape[0] // 4
                predictions_splits = []
                loss_splits = []
                
                for j in range(4):
                    start_idx = j * split_size
                    end_idx = (j + 1) * split_size if j < 3 else None
                    
                    video_split = batch_video_tensor[start_idx:end_idx]
                    audio_split = batch_audio_tensor[start_idx:end_idx]
                    labels_split = batch_labels_tensor[start_idx:end_idx]
                    
                    # 네트워크의 forward pass 호출
                    logits_split = network(video_split, audio_split)
                    
                    # 손실 및 예측 계산
                    loss_split = criterion(logits_split, labels_split)
                    total_loss += loss_split.item()
                    
                    predictions_split = torch.sigmoid(logits_split)
                    predictions_splits.append(predictions_split)
                    
                predictions = torch.cat(predictions_splits, dim=0)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_labels_tensor.cpu().numpy())
                
                if args.system == "windows":
                    key = key.replace("/", "\\")
                    sep = "\\"
                else:
                    sep = "/"

                if args.FTP:
                    path = key
                else:
                    parts = key.split(sep)
                    path_relative = os.path.join(*parts[-4:])

                # 예측 결과를 저장할 디렉터리 경로를 만듭니다.
                predictions_dir = "data"
                output_dir = os.path.dirname(os.path.join(predictions_dir, path_relative))
                predictions_output_dir = os.path.join(output_dir, os.path.splitext(os.path.split(key)[1])[0])

                # 디렉터리가 없으면 생성합니다.
                os.makedirs(predictions_output_dir, exist_ok=True)

                # 변경된 경로로 파일을 저장합니다.
                predictions_name = os.path.join(predictions_output_dir, f"{args.output}_{os.path.splitext(os.path.split(key)[1])[0]}.npy")

                np.save(predictions_name, predictions.cpu().numpy())

            except Exception as e:
                tqdm.write(f"⚠️ 경고: {key} 데이터 처리 중 오류 발생. 건너뜁니다. 오류: {e}")
                continue  # 다음 반복으로 넘어갑니다.

    # 전체 테스트 데이터에 대한 최종 메트릭 계산
    all_predictions_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    # mAP 계산 (class 1, 2, 3)
    aps = []
    for i in range(dataset.num_classes):
        try:
            ap = average_precision_score(all_labels_np[:, i], all_predictions_np[:, i])
        except ValueError:
            ap = 0.0
        aps.append(ap)
    
    mean_ap = np.mean([aps[1], aps[2], aps[3]])
    
    # Confusion Matrix 및 Accuracy 계산
    preds_argmax = np.argmax(all_predictions_np, axis=1)
    labels_argmax = np.argmax(all_labels_np, axis=1)
    
    cm = confusion_matrix(labels_argmax, preds_argmax, labels=range(dataset.num_classes))
    
    good_sample = np.diag(cm)
    bad_sample = np.sum(cm, axis=1) - good_sample
    accuracies = good_sample / (bad_sample + good_sample + 1e-10)
    accuracy = np.mean(accuracies)
    
    # 평균 손실 계산
    avg_loss = total_loss / len(all_predictions)

    print(cm)
    print(('auc: %.3f   (auc_PR_0: %.3f auc_PR_1: %.3f auc_PR_2: %.3f auc_PR_3: %.3f)') %
        (np.mean(aps), aps[0], aps[1], aps[2], aps[3]))
    print(' Loss: {:<8.3} Accuracy: {:<5.3} mAP: {:<5.3}'.format(avg_loss, accuracy, mean_ap))
    print(' Time: {:<8.3} s'.format(time.time()-start_time))

    if args.FTP:
        dataset.remove_files()

def create_directory(dir_name):
    """Creates a directory if it does not exist
    Parameters
    ----------
    dir_name : str
        The name of the directory to create
    """
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
        os.mkdir(dir_name)

if __name__ == "__main__":
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--testing', required=False, type=str, default='/media/giancos/Football/dataset_crop224/listgame_Test_2.npy', help='the file containg the testing data.')
    parser.add_argument('--featuresVideo', required=False, type=str, default="ResNET", help='select typeof features video')
    parser.add_argument('--featuresAudio', required=False, type=str, default="VGGish", help='select typeof features audio')
    parser.add_argument('--architecture', required=True, type=str, help='the name of the architecture to use.')
    parser.add_argument('--network', required=False, type=str, default="RVLAD", help='Select the type of network (CNN, MAX, AVERAGE, VLAD)')
    parser.add_argument('--VLAD_k', required=False, type=int, default=64, help='number of cluster for slustering method (NetVLAD, NetRVLAD, NetDBOW, NetFV)' )
    parser.add_argument('--WindowSize', required=False, type=int, default=60, help='Size of the Window' )
    parser.add_argument('--output', required=False, type=str, default="", help="Prefix for the output files.")
    parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use' )
    parser.add_argument('--PCA', required=False, action="store_true", help='use PCA version of the features')
    parser.add_argument('--tflog', required=False, type=str, default='Model', help='folder for tensorBoard output')
    parser.add_argument('--loglevel', required=False, type=str, default='INFO', help='logging level')
    parser.add_argument('--system', required=False, type=str, default="linux", help='linux or windows')
    parser.add_argument('--FTP', required=False, action="store_true", help='use FTP version of the dataset')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    args.PCA = True

    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=numeric_level)
    delattr(args, 'loglevel')
    if (args.GPU >= 0):
        if torch.cuda.is_available():
            torch.cuda.set_device(args.GPU)
        else:
            print("CUDA is not available, using CPU.")
    
    start=time.time()
    main(args)
    logging.info('Total Execution Time is {0} seconds'.format(time.time()-start))