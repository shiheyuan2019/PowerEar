import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import librosa.display

class AlignedDataset(BaseDataset):
   

    def __init__(self, opt):
        
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def getAB(self, AB_path):


        real_AB = librosa.load(AB_path, sr=16000)[0]
        crop_slice = 32385
        assert (len(real_AB) == crop_slice * 2)
        A = real_AB[:crop_slice]
        assert (len(A) == crop_slice)
        B = real_AB[crop_slice:]
        assert (len(B) == crop_slice)
        A_mag_spec = np.abs(librosa.stft(A, n_fft=510, hop_length=127))
        B_mag_spec = np.abs(librosa.stft(B, n_fft=510, hop_length=127))
        AB_mag_spec = np.hstack((A_mag_spec, B_mag_spec))
        AB_mag_spec_db = 20 * (np.log10(AB_mag_spec / 60 + 1e-6))
        if np.sum(AB_mag_spec_db >= 0) > 0:
            AB_mag_spec_db[AB_mag_spec_db > 0] = 0
        else:
            AB_mag_spec_db[
                np.argmax(AB_mag_spec_db) // AB_mag_spec_db.shape[1], np.argmax(AB_mag_spec_db) - AB_mag_spec_db.shape[
                    1] * (np.argmax(AB_mag_spec_db) // AB_mag_spec_db.shape[1])] = 0
        if np.sum(AB_mag_spec_db <= -80) > 0:
            AB_mag_spec_db[AB_mag_spec_db < -80] = -80
        else:
            AB_mag_spec_db[
                np.argmin(AB_mag_spec_db) // AB_mag_spec_db.shape[1], np.argmin(AB_mag_spec_db) - AB_mag_spec_db.shape[
                    1] * (np.argmin(AB_mag_spec_db) // AB_mag_spec_db.shape[1])] = -80

        AB_mag_spec_db_img = 255 * (AB_mag_spec_db - np.min(AB_mag_spec_db)) / (
                    np.max(AB_mag_spec_db) - np.min(AB_mag_spec_db))

        AB = Image.fromarray(AB_mag_spec_db_img.astype('uint8')).convert('RGB')
        return AB
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        AB_path = self.AB_paths[index]
        AB = self.getAB(AB_path)
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
