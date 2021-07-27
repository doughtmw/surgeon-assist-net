import torch
import time
from collections import OrderedDict


# Using ryzen 3600 for benchmark (single core) compare to snapdragon 850 on HL2
# https://gadgetversus.com/processor/qualcomm-sdm850-snapdragon-850-vs-amd-ryzen-5-3600/
def test_cpu_inference(test_generator, model, params, best_model_str=None):
    # Force batch size to 1 for inference time computations
    params['batch_size'] = 1
    CPU = torch.device('cpu')

    batch_progress = 0
    total_time = 0
    frame_count = 0
    test_acc = 0
    test_num_of_images = 0

    # Using cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum')

    # Get the total length of the training and validation dataset, use 500 samples
    total_num_test = 500  # use a short sequence of test data for speed (will not replicate the actual accuracy)
    print('total_num_test:', total_num_test)

    all_preds = []
    all_gts = []

    # To CPU
    model.to(CPU)

    # Load the best weights from training if not none
    if best_model_str != None:
        state_dict = torch.load(best_model_str)

        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)

    # Testing inference speed on CPU
    model.eval()
    with torch.no_grad():
        for local_batch, local_labels in test_generator:

            # If we have reached frame count, break
            if frame_count >= total_num_test:
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
            for i in range(len(preds)):
                all_preds.append(preds[i])
                all_gts.append(local_labels.data[i])

                # Increment counter
                frame_count += 1


            # Increment correct result count and loss
            test_acc += torch.sum(preds == local_labels.data)
            total_time += time_diff
            test_num_of_images += total_num_test

            batch_progress += 1

            # If we have reached the end of a batch
            if batch_progress * params['batch_size'] >= total_num_test:
                percent = 100.0
                print('\rTest progress: {:.2f} % [{} / {}]'.format(
                    percent,
                    total_num_test,
                    total_num_test),
                    end='\n')
            else:
                percent = batch_progress * params['batch_size'] / total_num_test * 100
                print('\rTest progress: {:.2f} % [{} / {}]'.format(
                    percent,
                    batch_progress * params['batch_size'],
                    total_num_test),
                    end='')

            # Each loop inference time
            # print('Inference time: {:.3f}ms'.format(time_diff / params['batch_size'] * 1000))

        test_acc = float(test_acc) / total_num_test

        print('Average inference: [{:.3f} ms] Acc: [{:.4f}]'.format(
            (total_time / (batch_progress * params['batch_size']))*1000,
            test_acc))
            