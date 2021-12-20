import torch
# from image_synthesis.data.base_dataset import ConcatDatasetWithIndex as ConcatDataset
from torch.utils.data import ConcatDataset
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.distributed.distributed import is_distributed

def build_dataloader(config, args=None, return_dataset=False):
    dataset_cfg = config['dataloader']
    train_dataset = []
    for ds_cfg in dataset_cfg['train_datasets']:
        ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')
        ds = instantiate_from_config(ds_cfg)
        train_dataset.append(ds)
    if len(train_dataset) > 1:
        train_dataset = ConcatDataset(train_dataset)
    else:
        train_dataset = train_dataset[0]
    
    val_dataset = []
    for ds_cfg in dataset_cfg['validation_datasets']:
        ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')
        ds = instantiate_from_config(ds_cfg)
        val_dataset.append(ds)
    if len(val_dataset) > 1:
        val_dataset = ConcatDataset(val_dataset)
    else:
        val_dataset = val_dataset[0]
    
    if args is not None and args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        train_iters = len(train_sampler) // dataset_cfg['batch_size']
        val_iters = len(val_sampler) // dataset_cfg['batch_size']
    else:
        train_sampler = None
        val_sampler = None
        train_iters = len(train_dataset) // dataset_cfg['batch_size']
        val_iters = len(val_dataset) // dataset_cfg['batch_size']

    # if args is not None and not args.debug:
    #     num_workers = max(2*dataset_cfg['batch_size'], dataset_cfg['num_workers'])
    #     num_workers = min(64, num_workers)
    # else:
    #     num_workers = dataset_cfg['num_workers']
    num_workers = dataset_cfg['num_workers']
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=dataset_cfg['batch_size'], 
                                               shuffle=(train_sampler is None),
                                               num_workers=num_workers, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True,
                                               persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=dataset_cfg['batch_size'], 
                                             shuffle=False, #(val_sampler is None),
                                             num_workers=num_workers, 
                                             pin_memory=True, 
                                             sampler=val_sampler, 
                                             drop_last=True,
                                             persistent_workers=True)

    dataload_info = {
        'train_loader': train_loader,
        'validation_loader': val_loader,
        'train_iterations': train_iters,
        'validation_iterations': val_iters
    }
    
    if return_dataset:
        dataload_info['train_dataset'] = train_dataset
        dataload_info['validation_dataset'] = val_dataset

    return dataload_info
