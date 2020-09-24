import os
import glob
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset for multiple dicom images.
    Args:
        dataset: [str] path of dicom images (should be a csv or txt file)
        transform: transformation function
    """

    def __init__(self, dataset, transform):

        self.dataset_folder = os.path.dirname(dataset)
        
        if os.path.splitext(dataset)[1] == '.csv':
            self.dataset = np.array(pd.read_csv(dataset).get("data"))
        elif os.path.splitext(dataset)[1] == '.txt':
            with open(dataset, 'r') as f:
                self.dataset = f.read().splitlines()
        else:
            raise ValueError("Wrong dataset format !")

        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        batch = Reader.read_multiple_dicom(os.path.join(self.dataset_folder, self.dataset[index]))

        batch = self.transform(batch)
        shape = batch["data"].shape
        batch["data"] = batch["data"].view(shape[0], -1, shape[-2], shape[-1])

        return batch


class Reader(object):

    @classmethod
    def read_multiple_dicom(cls, filenames, data_key="data", keywords=[]):
        '''
        Read multiple dicom files (follow the keywords defined in keyword list) in a folder
        Note:
            This method assumes that the folder contains all the slices for a given volume.
            Then, it will add new axis at first dimension (expand channel dimension)
        Args:
            filenames / folder: [str] or [list of str],
                - the folder for a case, which contains the case's dicom files
                - the dicom files for a case
            keywords: [list of str] => each <str> represents the keyword to select dicom files in folder
        Returns:
            data_dict: <dict>
                - "data": <ndarray> or - "mask": <ndarray>
        '''

        def load_all_dicom_images(fnames_p):

            series_instance_uid = None
            study_instance_uid = None

            images = []
            fnames = glob.glob(os.path.join(fnames_p, '*.dcm'))

            for fname in fnames:
                image = pydicom.dcmread(fname)

                if 'rt' in image.Modality.lower():
                    continue

                seid = str(image.SeriesInstanceUID).strip()
                stid = str(image.StudyInstanceUID).strip()

                if series_instance_uid is None:
                    series_instance_uid = seid
                    study_instance_uid = stid
                else:
                    if seid != series_instance_uid or stid != study_instance_uid:
                        print('*** skipped unmatched dicom file')
                        continue

                images.append(image)

            zs = [float(img.ImagePositionPatient[-1]) for img in images]
            sort_inds = np.argsort(zs)[::-1]
            images = [images[s] for s in sort_inds]

            return images

        def to_volume(images):
            """
            :param images:
            :return: numpy array of (z, y, x)
            """

            volume = np.stack([s.pixel_array for s in images])
            volume = volume.astype(np.float32)

            for slice_number in range(len(images)):
                if "RescaleType" in images[slice_number] and (images[slice_number].RescaleType not in ["HU"]):
                    if "RescaleSlope" in images[slice_number]:
                        slope = images[slice_number].RescaleSlope
                        if slope != 1:
                            volume[slice_number] = slope * volume[slice_number].astype(np.float64)
                            volume[slice_number] = volume[slice_number].astype(np.int16)
                    if "RescaleIntercept" in images[slice_number]:
                        intercept = images[slice_number].RescaleIntercept
                        volume[slice_number] += np.int16(intercept)

            volume[volume < 0] = 0

            return images[0].SeriesInstanceUID, volume

        images = load_all_dicom_images(filenames)
        seriesID, volume = to_volume(images)

        volume = np.expand_dims(volume, axis=0)

        volume_size = volume.shape[1:]

        return {data_key: volume, data_key + "_zhw": volume_size, data_key + "_seriesID": seriesID}
