import torch
import torch.nn as nn
import time

global_epoch = 0

# class Grad_Accumulator():
#     def __init__(self, update_interval):
#         self.upd = update_interva

#     def update(self, loss, it):
#         loss = loss / self.upd
        


def epoch_train(model:nn.Module, epoch, device, data_loader, optimizer, scheduler, logger, saver, update_interval):
    model.train()
    global global_epoch
    global_epoch = epoch
    it = 0
    total_iter = data_loader.__len__()
    start_time = time.time()
    logger.new_epoch()

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        it += 1
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        loginfo = {
            'mode': "train",
            'datatime': round(time.time() - start_time, 4),
            'epoch': epoch,
            'iter': it,
            'lr': optimizer.param_groups[0]['lr'],
            'total': total_iter,
            'batchsize': targets.size(0)
        }
    
        outputs = model(inputs, targets)
        
        loss = outputs['loss']

        if update_interval > 1:
            loss = loss / update_interval
        
        loss.backward()

        if it % update_interval == 0:
            optimizer.step()
            optimizer.zero_grad()

        loginfo['time'] = round(time.time() - start_time, 4)
        start_time = loginfo['time'] + start_time
        logger.record_train(loginfo, outputs)
    
    if it % update_interval != 0:
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    saver.save(epoch, model, optimizer)

def epoch_train_multigpu(model:nn.Module, epoch, device, data_loader, optimizer, scheduler, logger, saver, update_interval):
    model.train()
    global global_epoch
    global_epoch = epoch
    it = 0
    total_iter = data_loader.__len__()
    start_time = time.time()
    logger.new_epoch()

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        it += 1
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        loginfo = {
            'mode': "train",
            'datatime': round(time.time() - start_time, 4),
            'epoch': epoch,
            'iter': it,
            'lr': optimizer.param_groups[0]['lr'],
            'total': total_iter,
            'batchsize': targets.size(0)
        }

        if it % update_interval == 0 or it == total_iter:
            outputs = model(inputs, targets)
            loss = outputs['loss']

            if update_interval > 1:
                loss = loss / update_interval
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with model.no_sync():
                outputs = model(inputs, targets)
                loss = outputs['loss']

                if update_interval > 1:
                    loss = loss / update_interval
            
                loss.backward()

        loginfo['time'] = round(time.time() - start_time, 4)
        start_time = loginfo['time'] + start_time
        logger.record_train(loginfo, outputs)

    scheduler.step()
    saver.save(epoch, model, optimizer)

    torch.distributed.barrier()