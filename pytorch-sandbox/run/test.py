import torch
import os
import time
import numpy as np

# Local imports
from utils.plot_phase_results import plot_phase_results

def test(test_generator, best_model_str, model, params):
    frame_count = 0

    # Using cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum')

    # Get the total length of the training and validation dataset
    test_data_len = len(test_generator.dataset)
    print('test_data_len:', test_data_len)

    # Model computations
    batch_progress = 0
    test_acc = 0
    test_loss = 0
    test_num_of_images = 0
    test_start_time = time.time()

    all_preds = []
    all_gts = []

    # Transfer to device
    if params['use_cuda']:
        # Multiple GPU support
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model.to(params['device'])
    else:
        model.to(params['device'])

    # Load the best weights from training
    model.load_state_dict(torch.load(best_model_str))

    # Testing
    model.eval()
    with torch.no_grad():
        # folder to save to
        save_folder = best_model_str[:-34]
        save_folder = save_folder[8:]
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
                    os.makedirs('results/' + save_folder )

        # Append the ground truths to a file
        with open('results/' + save_folder + '/' + 'ground_truth.txt', 'a') as f:
            print('Frame', 'Phase', file=f)

        # Append the predictions to a results file
        with open('results/' + save_folder + '/' + 'pred.txt', 'a') as f:
            print('Frame', 'Phase', file=f)

        for local_batch, local_labels in test_generator:
            local_batch, local_labels = \
                local_batch.to(params['device']), \
                local_labels.to(params['device'])

            # Model inference to get output
            local_output = model(local_batch)
            # print('local_output:', local_output)

            # Reshape to [batch_size, n_classes]
            local_output = local_output[params['seq_len'] - 1::params['seq_len']]
            # print('local_output:', local_output)

            # Scale the raw output probabilities
            Sm = torch.nn.Softmax()
            local_output = Sm(local_output)
            # print('local_output_softmax:', local_output)

            # Get max value of model as prediction
            possibility, preds = torch.max(local_output.data, 1)
            # print('preds:', preds, 'local_labels:', local_labels)
            # print('possibility:', possibility)

            # Append the predictions and labels
            for i in range(len(preds)):
                all_preds.append(preds[i])
                all_gts.append(local_labels.data[i])

                # Append the ground truths to a file
                with open('results/' + save_folder + '/' + 'ground_truth.txt', 'a') as f:
                    print(frame_count, params['target_names'][local_labels.data[i]], file=f)

                # Append the predictions to a results file
                with open('results/' + save_folder + '/' + 'pred.txt', 'a') as f:
                    print(frame_count, params['target_names'][preds[i]], file=f)

                # Increment counter
                frame_count += 1

            loss = loss_fn(local_output, local_labels)

            # Increment correct result count and loss
            test_acc += torch.sum(preds == local_labels.data)
            test_loss += loss.data.item()
            test_num_of_images += len(local_labels)

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

        test_elapsed_time = time.time() - test_start_time
        test_acc = float(test_acc) / test_num_of_images
        test_loss = float(test_loss) / test_num_of_images

    # Print stats for train and validation for this epoch
    print('Time: [{} m {:.2f} s] '
          'Acc: [{:.4f}] '
          'Loss: [{:.4f}]'.format(
        test_elapsed_time // 60, test_elapsed_time % 60,
        test_acc,
        test_loss))

    # Plot a figure of the phase prediction results
    plot_phase_results('results/' + save_folder + '/', 'ground_truth.txt', 'pred.txt')

    # Format the predictions
    # y_true = np.asarray(all_gts, dtype=np.float32)
    # y_pred = np.asarray(all_preds, dtype=np.float32)