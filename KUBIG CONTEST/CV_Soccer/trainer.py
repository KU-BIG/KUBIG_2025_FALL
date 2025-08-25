import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import time
from sklearn.metrics import average_precision_score, confusion_matrix

class Trainer():
    def __init__(self, dataset, network, video_network=None, audio_network=None, output_prefix=""):
        self.network = network
        self.dataset = dataset
        self.output_prefix = output_prefix + "_" + str(date.today().isoformat() + time.strftime('_%H-%M-%S'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def _calculate_metrics(self, all_labels, all_preds):
        """Calculates metrics from all predictions and labels."""
        # 이 함수는 TensorFlow 코드의 메트릭 계산 로직을 모방합니다.
        
        # Calculate AUC PR (mAP)
        # sklearn의 average_precision_score를 사용하여 각 클래스별 AP를 계산
        # 0_background는 제외하고 1, 2, 3 클래스에 대한 mAP를 계산
        all_labels_np = np.concatenate(all_labels, axis=0)
        all_preds_np = np.concatenate(all_preds, axis=0)

        aps = []
        for i in range(self.dataset.num_classes):
            try:
                ap = average_precision_score(all_labels_np[:, i], all_preds_np[:, i])
                aps.append(ap)
            except ValueError:
                # 라벨이 하나만 있는 경우(e.g., all 0s or all 1s), AP 계산 불가
                aps.append(0.0)

        mAP = np.mean([aps[i] for i in range(1, self.dataset.num_classes)])
        
        # Calculate Accuracy and Confusion Matrix
        # argmax를 사용하여 예측값을 이진(binary)에서 다중 클래스(multiclass)로 변환
        preds_argmax = np.argmax(all_preds_np, axis=1)
        labels_argmax = np.argmax(all_labels_np, axis=1)

        cm = confusion_matrix(labels_argmax, preds_argmax, labels=range(self.dataset.num_classes))
        
        good_sample = np.diag(cm)
        bad_sample = np.sum(cm, axis=1) - good_sample
        accuracies = good_sample / (bad_sample + good_sample + 1e-10)
        accuracy = np.mean(accuracies)

        # Loss는 배치별 평균을 사용해야 하므로 이 함수에서 계산하지 않습니다.
        
        return {
            'mAP': mAP,
            'auc_PR_0': aps[0],
            'auc_PR_1': aps[1],
            'auc_PR_2': aps[2],
            'auc_PR_3': aps[3],
            'auc_PR_4': aps[4], # 새로운 클래스 추가
            'accuracy': accuracy,
            'accuracies': accuracies,
            'confusion_matrix': cm
        }

    def train(self, epochs=1, learning_rate=0.001, tflog="logs"):
        self.tflog = tflog
        
        # SummaryWriters
        if not os.path.exists(self.tflog):
            os.makedirs(self.tflog)
        train_writer = SummaryWriter(os.path.join(self.tflog, self.output_prefix + "_training"))
        valid_writer = SummaryWriter(os.path.join(self.tflog, self.output_prefix + "_validation"))
        test_writer = SummaryWriter(os.path.join(self.tflog, self.output_prefix + "_testing"))
        
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.dataset.weights, device=self.device))
        
        best_validation_mAP = 0
        cnt_since_best_epoch = 0
        best_output_prefix = self.output_prefix
        
        previous_best_model_path = ""
        
        for epoch in range(epochs):
            start_time_epoch = time.time()
            
            print("\n\n\n")
            print(f"Epoch {epoch + 1:>2}: ")
            
            self.dataset.prepareNewEpoch()
            
            # --- Training Loop ---
            print("\nTraining")
            self.network.train()
            
            all_labels_train = []
            all_preds_train = []
            total_loss_train = 0
            
            start_time = time.time()
            for total_num_batches in range(self.dataset.nb_batch_training):
                batch_video_features, batch_audio_features, batch_labels, _ = self.dataset.getTrainingBatch(total_num_batches)
                
                # Move data to device
                video_input = torch.tensor(batch_video_features, dtype=torch.float32, device=self.device)
                audio_input = torch.tensor(batch_audio_features, dtype=torch.float32, device=self.device)
                labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                
                optimizer.zero_grad()
                logits = self.network(video_input, audio_input)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss_train += loss.item()
                preds = torch.sigmoid(logits)

                all_labels_train.append(labels.cpu().numpy())
                all_preds_train.append(preds.detach().cpu().numpy())
                
                if (total_num_batches + 1) % 10 == 0:
                    print(f"Batch number: {total_num_batches + 1:<3} Loss: {loss.item():.3f}")

            # Calculate train metrics
            vals_train = self._calculate_metrics(all_labels_train, all_preds_train)
            vals_train['loss'] = total_loss_train / self.dataset.nb_batch_training

            print(f"\nLoss: {vals_train['loss']:.3f} Accuracy: {vals_train['accuracy']:.3f} mAP: {vals_train['mAP']:.3f}")
            print(f"Time: {time.time()-start_time:.3f} s", flush=True)
            
            # Log training summaries
            train_writer.add_scalar('learning_rate', learning_rate, epoch)
            train_writer.add_scalar('loss', vals_train['loss'], epoch)
            train_writer.add_scalar('accuracy/average', vals_train['accuracy'], epoch)
            for i, acc in enumerate(vals_train['accuracies']):
                train_writer.add_scalar(f'accuracy/{i}', acc, epoch)
            train_writer.add_scalar('AP/mean', vals_train['mAP'], epoch)
            train_writer.add_text('confusion_matrix', str(vals_train['confusion_matrix']), epoch)
            
            # --- Validation Loop ---
            print("\nValidation")
            vals_valid = self.validate(criterion)
            
            # Log validation summaries
            valid_writer.add_scalar('learning_rate', learning_rate, epoch)
            valid_writer.add_scalar('loss', vals_valid['loss'], epoch)
            valid_writer.add_scalar('accuracy/average', vals_valid['accuracy'], epoch)
            for i, acc in enumerate(vals_valid['accuracies']):
                valid_writer.add_scalar(f'accuracy/{i}', acc, epoch)
            valid_writer.add_scalar('AP/mean', vals_valid['mAP'], epoch)
            valid_writer.add_text('confusion_matrix', str(vals_valid['confusion_matrix']), epoch)
            
            # Look for best model
            print("\nvalidation_mAP: " + str(vals_valid['mAP']))
            print("best_validation_mAP: " + str(best_validation_mAP))
            print("validation_mAP > best_validation_mAP ?: " + str(vals_valid['mAP'] > best_validation_mAP))
            print("cnt_since_best_epoch currently: " + str(cnt_since_best_epoch))
            print("elapsed time for this epoch: " + str(time.time() - start_time_epoch))
            
            if vals_valid['mAP'] > best_validation_mAP:
                # 새로운 최고 성능 모델을 찾았으므로 기존 파일을 삭제합니다.
                if previous_best_model_path and os.path.exists(previous_best_model_path):
                    try:
                        os.remove(previous_best_model_path)
                        print(f"🗑️ 이전 최고 성능 모델 파일 삭제: {previous_best_model_path}")
                    except OSError as e:
                        print(f"오류: 이전 모델 파일 삭제 실패: {e}")
                
                # 최고 성능 지표를 현재 값으로 업데이트합니다.
                best_validation_mAP = vals_valid['mAP']
                best_validation_accuracy = vals_valid["accuracy"]
                best_validation_loss = vals_valid['loss']
                best_epoch = epoch
                cnt_since_best_epoch = 0
                
                # 새로운 최고 성능 모델 가중치를 저장합니다.
                best_output_prefix = self.output_prefix + f"_epoch_{epoch}"
                current_best_model_path = os.path.join(self.tflog, best_output_prefix + "_model.pth")
                
                torch.save(self.network.state_dict(), current_best_model_path)
                print(f"💾 새로운 최고 성능 모델 저장: {current_best_model_path}")

                # 다음 에포크를 위해 현재 저장된 모델 경로를 기록합니다.
                previous_best_model_path = current_best_model_path
            else:
                cnt_since_best_epoch += 1

            if cnt_since_best_epoch > 10 and learning_rate > 0.0001:
                print("reducing LR after plateau since " + str(cnt_since_best_epoch) + " epochs without improvements")
                learning_rate /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                cnt_since_best_epoch = 0
                # Restore best model weights
                self.network.load_state_dict(torch.load(os.path.join(self.tflog, best_output_prefix + "_model.pth")))
            elif cnt_since_best_epoch > 30:
                print("stopping after plateau since " + str(cnt_since_best_epoch) + " epochs without improvements")
                break

        train_writer.close()
        valid_writer.close()
        print("stopping after " + str(epoch) + " epochs maximum training reached")

        # --- Testing Loop ---
        print("\nTesting")
        # Restore best model weights
        self.network.load_state_dict(torch.load(os.path.join(self.tflog, best_output_prefix + "_model.pth")))
        vals_test = self.test(criterion)

        test_writer.add_scalar('loss', vals_test['loss'])
        test_writer.add_scalar('accuracy/average', vals_test['accuracy'])
        for i, acc in enumerate(vals_test['accuracies']):
            test_writer.add_scalar(f'accuracy/{i}', acc)
        test_writer.add_scalar('AP/mean', vals_test['mAP'])
        test_writer.add_text('confusion_matrix', str(vals_test['confusion_matrix']))
        test_writer.close()

        return vals_train, vals_valid, vals_test, best_output_prefix

    def validate(self, criterion):
        self.network.eval()
        
        all_labels_valid = []
        all_preds_valid = []
        total_loss_valid = 0
        
        start_time = time.time()
        with torch.no_grad():
            for i in range(self.dataset.nb_batch_validation):
                batch_video_features, batch_audio_features, batch_labels = self.dataset.getValidationBatch(i)
                
                video_input = torch.tensor(batch_video_features, dtype=torch.float32, device=self.device)
                audio_input = torch.tensor(batch_audio_features, dtype=torch.float32, device=self.device)
                labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                
                logits = self.network(video_input, audio_input)
                loss = criterion(logits, labels)
                
                total_loss_valid += loss.item()
                preds = torch.sigmoid(logits)

                all_labels_valid.append(labels.cpu().numpy())
                all_preds_valid.append(preds.detach().cpu().numpy())

        vals_valid = self._calculate_metrics(all_labels_valid, all_preds_valid)
        vals_valid['loss'] = total_loss_valid / self.dataset.nb_batch_validation

        print(vals_valid['confusion_matrix'])
        print(f"Loss: {vals_valid['loss']:.3f} Accuracy: {vals_valid['accuracy']:.3f} mAP: {vals_valid['mAP']:.3f}")
        print(f"Time: {time.time()-start_time:.3f} s")
        
        return vals_valid

    def test(self, criterion):
        self.network.eval()
        
        all_labels_test = []
        all_preds_test = []
        total_loss_test = 0
        
        start_time = time.time()
        with torch.no_grad():
            for i in range(self.dataset.nb_batch_testing):
                batch_video_features, batch_audio_features, batch_labels = self.dataset.getTestingBatch(i)
                
                video_input = torch.tensor(batch_video_features, dtype=torch.float32, device=self.device)
                audio_input = torch.tensor(batch_audio_features, dtype=torch.float32, device=self.device)
                labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                
                logits = self.network(video_input, audio_input)
                loss = criterion(logits, labels)
                
                total_loss_test += loss.item()
                preds = torch.sigmoid(logits)

                all_labels_test.append(labels.cpu().numpy())
                all_preds_test.append(preds.detach().cpu().numpy())
        
        vals_test = self._calculate_metrics(all_labels_test, all_preds_test)
        vals_test['loss'] = total_loss_test / self.dataset.nb_batch_testing

        print(vals_test['confusion_matrix'])
        print(f"Loss: {vals_test['loss']:.3f} Accuracy: {vals_test['accuracy']:.3f} mAP: {vals_test['mAP']:.3f}")
        print(f"Time: {time.time()-start_time:.3f} s")
        
        return vals_test

    def predict(self, prop, display=True, tflog="logs"):
        self.tflog = tflog
        
        if not os.path.exists(self.tflog):
            os.makedirs(self.tflog)
        
        # Assuming the model path is provided in the network initialization
        # self.network.initialize()

        self.network.eval()
        
        all_labels_test = []
        all_preds_test = []
        total_loss_test = 0
        prop_l = []

        start_time = time.time()
        with torch.no_grad():
            for i in range(self.dataset.nb_batch_testing):
                batch_video_features, batch_audio_features, batch_labels = self.dataset.getTestingBatch(i)
                
                video_input = torch.tensor(batch_video_features, dtype=torch.float32, device=self.device)
                audio_input = torch.tensor(batch_audio_features, dtype=torch.float32, device=self.device)
                labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)

                # Assuming BCEWithLogitsLoss for consistency
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.dataset.weights, device=self.device))
                
                logits = self.network(video_input, audio_input)
                loss = criterion(logits, labels)

                total_loss_test += loss.item()
                preds = torch.sigmoid(logits)

                all_labels_test.append(labels.cpu().numpy())
                all_preds_test.append(preds.detach().cpu().numpy())

                # Get prediction/logits
                prop_output = getattr(self.network, prop)
                prop_l.append(prop_output.detach().cpu().numpy())

        vals_test = self._calculate_metrics(all_labels_test, all_preds_test)
        vals_test['loss'] = total_loss_test / self.dataset.nb_batch_testing

        prop_l = np.concatenate(prop_l, axis=0)
        np.save(os.path.join(self.tflog, prop), prop_l)
        
        if display:
            print(vals_test['confusion_matrix'])
            print(f"Loss: {vals_test['loss']:.3f} Accuracy: {vals_test['accuracy']:.3f} mAP: {vals_test['mAP']:.3f}")
            print(f"Time: {time.time()-start_time:.3f} s")

        return vals_test

    def predict_other(self):
        video_logits = np.load(self.network.video_input_file, allow_pickle=True)
        audio_logits = np.load(self.network.audio_input_file, allow_pickle=True)

        # Assuming BCEWithLogitsLoss for consistency
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.dataset.weights, device=self.device))
        
        self.network.eval()
        
        all_labels_test = []
        all_preds_test = []
        total_loss_test = 0
        
        start_time = time.time()
        with torch.no_grad():
            for i in range(self.dataset.nb_batch_testing):
                # getTestingBatch에서 video_features, audio_features, labels를 반환한다고 가정
                _, _, batch_labels, _ = self.dataset.getTestingBatch(i)

                video_input = torch.tensor(video_logits[i], dtype=torch.float32, device=self.device)
                audio_input = torch.tensor(audio_logits[i], dtype=torch.float32, device=self.device)
                labels = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)

                logits = self.network(video_input, audio_input)
                loss = criterion(logits, labels)
                
                total_loss_test += loss.item()
                preds = torch.sigmoid(logits)

                all_labels_test.append(labels.cpu().numpy())
                all_preds_test.append(preds.detach().cpu().numpy())
        
        vals_test = self._calculate_metrics(all_labels_test, all_preds_test)
        vals_test['loss'] = total_loss_test / self.dataset.nb_batch_testing

        print(vals_test['confusion_matrix'])
        print(f"Loss: {vals_test['loss']:.3f} Accuracy: {vals_test['accuracy']:.3f} mAP: {vals_test['mAP']:.3f}")
        print(f"Time: {time.time()-start_time:.3f} s")

        return vals_test