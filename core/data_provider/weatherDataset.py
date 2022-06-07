from scipy import linalg
from torch.utils.data import Dataset
from matplotlib.pyplot import  imread
import pandas as pd
import os
import numpy as np
import torch


def read_image(first20Path):
    return  np.asarray([imread(singleImg) for singleImg in first20Path])
def read_label(last20Path,key):
    return read_image(last20Path)
class CustomTrainImageDataset(Dataset):
    def __init__(self, allImagePath, factor,imgTransform=None, labelTransform=None):
        self.allImagePath = allImagePath
        self.imgTransform = imgTransform
        self.labelTransform = labelTransform
        self.factor = factor

    def __len__(self):
        return len(self.allImagePath)

    def __getitem__(self, idx):
        image = read_image(self.allImagePath[idx])/255*self.factor

        if self.imgTransform:
            image = self.imgTransform(np.expand_dims(image,0))


        return torch.tensor(image).squeeze(0)




class CustomValidImageDataset(Dataset):
    def __init__(self, allImagePath, factor,imgTransform=None, labelTransform=None):
        self.allImagePath = allImagePath
        self.imgTransform = imgTransform
        self.labelTransform = labelTransform
        self.factor = factor

    def __len__(self):
        return len(self.allImagePath)

    def __getitem__(self, idx):
        image = read_image(self.allImagePath[idx][0:20])/255*self.factor
        label = read_image(self.allImagePath[idx][20:])/255*self.factor

        if self.imgTransform:
            image = self.imgTransform(np.expand_dims(image,0))
            label = self.imgTransform(np.expand_dims(label,0))


        return torch.tensor(image).squeeze(0), torch.tensor(label).squeeze(0)




class CustomInferImageDataset(Dataset):
    def __init__(self, allImagePath, factor,imgTransform=None):
        self.allImagePath = allImagePath
        self.imgTransform = imgTransform
        self.factor =  factor

    def __len__(self):
        return len(self.allImagePath)

    def __getitem__(self, idx):
        image = readImgFromDir(self.allImagePath[idx],factor=self.factor)

        if self.imgTransform:
            image = self.imgTransform(np.expand_dims(image, 0))

        return torch.tensor(image).squeeze(0),self.allImagePath[idx]
def get_true_paths_from_csv(dataPath, csvPath):

    csvContent = pd.read_csv(csvPath,header=None)
    precip,radar,wind = [],[],[]
    for i in range(0,len(csvContent[1]),10):
        imagesSingleRow = list(csvContent.iloc[i].values)
        radars = list(map(lambda x:os.path.join(dataPath,"Radar","radar_"+x),imagesSingleRow))
        precips = list(map(lambda x:os.path.join(dataPath,"Precip","precip_"+x),imagesSingleRow))
        winds = list(map(lambda x:os.path.join(dataPath,"Wind","wind_"+x),imagesSingleRow))

        precip.append(np.asarray(precips))
        radar.append(np.asarray(radars))
        wind.append(np.asarray(winds))

    return {"precip":precip,"radar":radar,"wind":wind}

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def readImgFromDir(singleVolumeDir,factor):
    volume = np.zeros(shape=(20,480,560))
    allImgs = os.listdir(singleVolumeDir)
    # print(np.max(image),np.min(image))

    # image=np.clip(np.array(image),0,factorDict[key])/factorDict[key]*255.
    for i  in range(len(allImgs)):
        img = imread(os.path.join(singleVolumeDir,allImgs[i]))/255*factor
        volume[i,:,:] = img
    return volume
if __name__ == '__main__':
    a = np.random.random(size=(2, 20, 1, 480, 560))
    b = np.random.random(size=(2, 20, 1, 480, 560))
    mua = np.mean(a,0)
    mub = np.mean(a,0)
    print(a)