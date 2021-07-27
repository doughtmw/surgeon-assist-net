import torch
import time
import os
from collections import OrderedDict

# Local imports
from utils.convert_onnx import Onnx_Eval

# You may change the settings in this cell according to Performance Test Tool result.
# os.environ["OMP_NUM_THREADS"] = str(1) # 1 thread
# os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'


# Using ryzen 3600 for benchmark (single core) compare to snapdragon 850 on HL2
# https://gadgetversus.com/processor/qualcomm-sdm850-snapdragon-850-vs-amd-ryzen-5-3600/
def test_pytorch_onnx_accuracy(test_generator, model, params, save_name, best_model_str=None):
    params['batch_size'] = 1
    
    # Create a new instance of the onnx_eval class
    onnx_eval = Onnx_Eval(save_name)

    CPU = torch.device('cpu')

    batch_progress = 0
    total_time_pt = 0
    total_time_onnx = 0
    frame_count = 0
    test_acc_pt = 0
    test_acc_onnx = 0
    test_num_of_images = 0

    # Using cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum')

    # Get the total length of the training and validation dataset
    # test_data_len = len(test_generator.dataset)
    test_data_len = 500   # use a short sequence of test data for speed (will not replicate the actual accuracy)
    print('test_data_len:', test_data_len)

    all_preds = []
    all_gts = []

    # To CPU
    model.to(CPU)

    if best_model_str != None:
        # Load the best weights from training
        state_dict = torch.load(best_model_str)

        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)

    # Testing
    model.eval()
    with torch.no_grad():
        if best_model_str != None: 
            # folder to save to
            save_folder = best_model_str[:-34]
            save_folder = save_folder[8:]
            save_folder = 'pt_onnx_' + save_folder
            if not os.path.exists('results/' + save_folder):
                os.makedirs('results/' + save_folder)
            else:
                # if path exists, make folder at new path
                save_folder = save_folder + "_2"
                if not os.path.exists('results/' + save_folder):
                    os.makedirs('results/' + save_folder )
                else:
                    # if path exists, make folder at new path
                    save_folder = save_folder + "_3"
                    if not os.path.exists('results/' + save_folder):
                        os.makedirs('results/' + save_folder)

            # Append the ground truths to a file
            with open('results/' + save_folder + '/' + 'ground_truth.txt', 'a') as f:
                print('Frame', 'Phase', file=f)

            # Append the predictions to a results file
            with open('results/' + save_folder + '/' + 'pred_pt.txt', 'a') as f:
                print('Frame', 'Phase', file=f)

            # Append the predictions to a results file
            with open('results/' + save_folder + '/' + 'pred_onnx.txt', 'a') as f:
                print('Frame', 'Phase', file=f)

        for local_batch, local_labels in test_generator:

            # If we have reached frame count, break
            if frame_count >= test_data_len:
                break

            local_batch, local_labels = \
                local_batch.to(CPU), \
                local_labels.to(CPU)

            begin_time = time.time()

            # Model inference to get output
            local_output = model(local_batch)

            time_diff = time.time() - begin_time

            # Reshape to [batch_size, n_classes]
            local_output = local_output[params['seq_len'] - 1::params['seq_len']]

            # Scale the raw output probabilities
            Sm = torch.nn.Softmax()
            local_output = Sm(local_output)
            # print('local_output_softmax:', local_output)

            # Get max value of model as prediction
            possibility, preds = torch.max(local_output.data, 1)
            loss = loss_fn(local_output, local_labels)

            # Append the predictions and labels
            count = 0
            for i in range(len(preds)):
                all_preds.append(preds[i])
                all_gts.append(local_labels.data[i])

                # Get predictions from ONNX model
                onnx_input = local_batch[i].view(-1, 3, params['img_size'][0], params['img_size'][1]) 
                dict_onnx = onnx_eval.eval_pytorch_vs_onnx(onnx_input, save_name)
                time_diff_onnx = dict_onnx['onnx_time_diff']
                total_time_onnx += time_diff_onnx

                onnx_out_tensor = torch.from_numpy(dict_onnx['onnx_pred'])
                onnx_out_tensor = onnx_out_tensor[params['seq_len'] - 1::params['seq_len']]
                onnx_out_tensor = Sm(onnx_out_tensor)

                possiblity_onnx, preds_onnx = torch.max(onnx_out_tensor, 1)

                if best_model_str != None: 
                    # Append the ground truths to a file
                    with open('results/' + save_folder + '/' + 'ground_truth.txt', 'a') as f:
                        print(frame_count, params['target_names'][local_labels.data[i]], file=f)

                    # Append the predictions to a results file
                    with open('results/' + save_folder + '/' + 'pred_pt.txt', 'a') as f:
                        print(frame_count, params['target_names'][preds[i]], file=f)

                    # Append onnx predictions to file
                    with open('results/' + save_folder + '/' + 'pred_onnx.txt', 'a') as f:
                        print(frame_count, params['target_names'][preds_onnx], file=f)

                # Increment counter
                frame_count += 1


            # Increment correct result count and loss
            test_acc_pt += torch.sum(preds == local_labels.data)
            test_acc_onnx += torch.sum(preds_onnx.to(CPU) == local_labels.data)
            total_time_pt += time_diff
            test_num_of_images += test_data_len

            batch_progress += 1

            # If we have reached the end of a batch
            if batch_progress * params['batch_size'] >= test_data_len:
                percent = 100.0
                print('\rTest progress: {:.2f} % [{} / {}]'.format(
                    percent,
                    test_data_len,
                    test_data_len),
                    end='\n')
            else:
                percent = batch_progress * params['batch_size'] / test_data_len * 100
                print('\rTest progress: {:.2f} % [{} / {}]'.format(
                    percent,
                    batch_progress * params['batch_size'],
                    test_data_len),
                    end='')

            # Each loop inference time
            # print('Inference time: {:.3f}ms'.format(time_diff / params['batch_size'] * 1000))

        test_acc_pt = float(test_acc_pt) / test_data_len
        test_acc_onnx = float(test_acc_onnx) / test_data_len


        # Getting different accuracy between PT and ONNX implementations for batch size > 1, inference time is correct
        print('Average PT inference: [{:.3f} ms] PT Acc: [{:.4f}] Average ONNX inference: [{:.3f} ms] ONNX Acc: [{:.4f}]'.format(
            (total_time_pt / (batch_progress * params['batch_size']))*1000,
            test_acc_pt, 
            (total_time_onnx / (batch_progress * params['batch_size']))*1000,
            test_acc_onnx))
            