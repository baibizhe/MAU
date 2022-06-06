from torchvision import transforms
from torch.utils.data import DataLoader
from core.data_provider.mm import MovingMNIST
from core.data_provider.weatherDataset import CustomTrainImageDataset

#
# def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
#                   is_training=True,
#                   is_shuffle=True,):
#     if is_training:
#         num_workers = configs.num_workers
#         root = data_train_path
#     else:
#         num_workers = 0
#         root = data_test_path
#     dataset = MovingMNIST(is_train=is_training,
#                           root=root,
#                           n_frames=20,
#                           num_objects=[2])
#     return DataLoader(dataset,
#                       pin_memory=True,
#                       batch_size=batch_size,
#                       shuffle=is_shuffle,
#                       num_workers=num_workers)
# def getDataLoaders(args, dataPaths, key, trainTransform):
#     singleDataPaths = dataPaths[key]
#     splitIndex = int(len(singleDataPaths) * 0.8)
#     trainPaths = singleDataPaths[0:splitIndex]
#     validPaths = singleDataPaths[splitIndex:]
#     factorDict={"radar":70,"precip":35,"wind":10}
#     factor = factorDict[key]
#     radarTrainDataset = CustomTrainImageDataset(trainPaths, imgTransform=trainTransform,factor=factor)
#     train_loader = torch.utils.data.DataLoader(radarTrainDataset, batch_size=args.batch_size, num_workers=args.workers,
#                                                shuffle=True, prefetch_factor=4, pin_memory=True, drop_last=True)
#     radarValidDataset = CustomTrainImageDataset(validPaths, imgTransform=trainTransform,factor=factor)
#     valid_loader = torch.utils.data.DataLoader(radarValidDataset, batch_size=args.batch_size, num_workers=args.workers,
#                                                shuffle=False, prefetch_factor=4, pin_memory=True, drop_last=True)
#     print("train length {}  valid length {}".format(len(trainPaths), len(validPaths)))
#     return train_loader, valid_loader

def data_provider(configs, data_train_path, batch_size,key,
                  train_transform=None,
                  valid_transform=None,
                  is_training=True,
                  is_shuffle=True):
    factorDict={"radar":70,"precip":35,"wind":10}
    factor = factorDict[key]
    splitIndex = int(len(data_train_path) * 0.8)
    train_paths = data_train_path[0:splitIndex]
    valid_oaths = data_train_path[splitIndex:]
    if is_training:
        num_workers = configs.num_workers
        dataset = CustomTrainImageDataset(train_paths, imgTransform=train_transform, factor=factor)

    else:
        num_workers = 0
        dataset = CustomTrainImageDataset(valid_oaths, imgTransform=train_transform, factor=factor)

    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)

