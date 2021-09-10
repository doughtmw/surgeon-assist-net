import torch
import time
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np


def train(training_generator, validation_generator, model, params):
    # Create tensorboard logger
    writer_name = "runs/" \
                  + "_img_size_" + str(params['img_size']) \
                  + "_length_" + str(params['seq_len']) \
                  + "_hidden_size_" + str(params['hidden_size'])
    writer = SummaryWriter(writer_name)

    # Using cross entropy loss function
    print("Using weighting to rebalance classes:",params['class_weight'])
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum',
        weight=params['class_weight'].to(params['device'], dtype=torch.float))

    # Specify per-layer learning rates of parameters for SGD optimizer
    if (params["other_model"]) is None:
        optimizer = torch.optim.SGD([
            {'params': model.feat.parameters(), 'lr': params['learning_rate'] / 10},
            {'params': model.rnn.parameters()},
            {'params': model.pred.parameters()}],
            lr=params['learning_rate'],
            momentum=params['momentum'],
            weight_decay=params['learning_rate'])
        print("Using layer-wise SGD.")

    else:
        # For testing other models do not use layer-wise optimization
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            momentum=params['momentum'],
            weight_decay=params['learning_rate'])
        print("Using regular SGD.")

    # Reduce LR on training plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)

    # Get the total length of the training and validation dataset
    train_data_len = len(training_generator.dataset)
    val_data_len = len(validation_generator.dataset)
    print('train_data_len:', train_data_len)
    print('val_data_len:', val_data_len)

    # Cache the validation accuracy of best model
    best_model_val_acc = 0

    # Transfer to device
    if params['use_cuda']:
        # Multiple GPU support
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.to(params['device'])
    else:
        model.to(params['device'])

    # Loop over epochs
    for epoch in range(params['epochs']):

        # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
        np.random.seed() # reset seed

        batch_progress = 0
        train_acc = 0
        train_loss = 0
        train_num_of_images = 0
        train_start_time = time.time()

        # Training
        model.train()
        for local_batch, local_labels in training_generator:
            optimizer.zero_grad()

            # Transfer data to GPU
            local_batch, local_labels = \
                local_batch.to(params['device']), \
                local_labels.to(params['device'])

            # Model inference to get output
            local_output = model(local_batch)

            # Reshape to [batch_size, n_classes]
            local_output = local_output[params['seq_len'] - 1::params['seq_len']]

            # Debug
            # print('local_batch.shape:', local_batch.shape)
            # print('local_output.shape:', local_output.shape)
            # print('local_labels.shape:', local_labels.shape)
            # print('local_labels:', local_labels)

            # Get max value of model as prediction
            _, preds = torch.max(local_output.data, 1)
            # print('preds:', preds)

            # Compute loss at step
            loss = loss_fn(local_output, local_labels)
            # print('loss:', loss)

            # Backward pass, next optimizer step
            loss.backward()
            optimizer.step()

            # Increment correct result count and loss
            batch_corrects = torch.sum(preds == local_labels.data)
            train_acc += batch_corrects
            train_loss += loss.data.item()
            train_num_of_images += len(local_labels)

            # batch_acc = float(batch_corrects) / (params['batch_size'] * params['seq_len'])
            batch_acc = float(batch_corrects) / (params['batch_size'])
            batch_loss = float(loss.data.item()) / (params['batch_size'])
            batch_progress += 1

            # If we have reached the end of a batch
            if batch_progress * params['batch_size'] >= train_data_len:
                percent = 100.0
                print('\rBatch progress: {:.2f} % [{} / {}] Batch loss: {:.2f} Batch acc: {:.2f}'.format(
                    percent,
                    train_data_len,
                    train_data_len,
                    batch_loss,
                    batch_acc,),
                    end='\n')
            else:
                percent = batch_progress * params['batch_size'] / train_data_len * 100
                print('\rBatch progress: {:.2f} % [{} / {}] Batch loss: {:.2f} Batch acc: {:.2f}'.format(
                    percent,
                    batch_progress * params['batch_size'],
                    train_data_len,
                    batch_loss,
                    batch_acc),
                    end='')

        # Get elapsed time, accuracy and loss for epoch
        train_elapsed_time = time.time() - train_start_time
        train_acc = float(train_acc) / train_num_of_images
        train_loss = float(train_loss) / train_num_of_images

        # Model computations
        batch_progress = 0
        val_acc = 0
        val_loss = 0
        val_num_of_images = 0
        val_start_time = time.time()

        # Validation
        model.eval()
        with torch.no_grad():
            for local_batch, local_labels in validation_generator:

                local_batch, local_labels = \
                    local_batch.to(params['device']), \
                    local_labels.to(params['device'])

                # Model inference to get output
                local_output = model(local_batch)

                # Reshape to [batch_size, n_classes]
                local_output = local_output[params['seq_len'] - 1::params['seq_len']]

                # Get max value of model as prediction
                possibility, preds = torch.max(local_output.data, 1)
                # print('preds:', preds, 'local_labels:', local_labels)
                # print('possibility:', possibility)

                # Compute loss at step
                loss = loss_fn(local_output, local_labels)

                # Increment correct result count and loss
                val_acc += torch.sum(preds == local_labels.data)
                val_loss += loss.data.item()
                val_num_of_images += len(local_labels)

                batch_progress += 1

                # If we have reached the end of a batch
                if batch_progress * params['batch_size'] >= val_data_len:
                    percent = 100.0
                    print('\rVal progress: {:.2f} % [{} / {}]'.format(
                        percent,
                        val_data_len,
                        val_data_len),
                        end='\n')
                else:
                    percent = batch_progress * params['batch_size'] / val_data_len * 100
                    print('\rVal progress: {:.2f} % [{} / {}]'.format(
                        percent,
                        batch_progress * params['batch_size'],
                        val_data_len),
                        end='')

            val_elapsed_time = time.time() - val_start_time
            val_acc = float(val_acc) / val_num_of_images
            val_loss = float(val_loss) / val_num_of_images

            # Step scheduler forward using MixedAveragePointDistanceMean_in_mm
            scheduler.step(val_loss)

        # Print stats for train and validation for this epoch
        print('Epoch: {} '
              'Time: [{} m {:.2f} s | {} m {:.2f} s] '
              'Acc: [{:.3f} | {:.3f}] '
              'Loss: [{:.2f} | {:.2f}]'.format(
            epoch,
            train_elapsed_time // 60, train_elapsed_time % 60,
            val_elapsed_time // 60, val_elapsed_time % 60,
            train_acc, val_acc,
            train_loss, val_loss))

        # Check to see if the model is best performing on validation dataset
        if val_acc > best_model_val_acc:
            best_model_val_acc = val_acc

            # Get model weights
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_str = '{}{}__feat_{}__img_{}__len_{}__hsize_{}__epo_{}__train_{:.3f}__val_{:.3f}__best.pth'.format(
                params['weights_dir'],
                params['dataset'],
                params['feat_ext'],
                params['img_size'][0],
                params['seq_len'],
                params['hidden_size'],
                epoch,
                train_acc, val_acc)

            # Save only the model parameters
            torch.save(best_model_wts, best_model_str)
            print('Saved model: {} Best epoch: {} Best acc: [{:.3f} | {:.3f}]\n'.format(
                best_model_str,
                epoch,
                train_acc,
                val_acc))
        else:
            print('')

        # log training stats to tensorboard
        writer.add_scalar('training accuracy', float(train_acc), epoch)
        writer.add_scalar('training loss', float(train_loss), epoch)
        writer.add_scalar('validation accuracy', float(val_acc), epoch)
        writer.add_scalar('validation loss', float(val_loss), epoch)

    print('Done training model.\n')
    return best_model_str
