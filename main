# Create DataSet and Transform to read medical data, convert to numpy, normalize, and resample to appropriate size
# This part of the code directly fine tunes medicalnet and serves as the baseline for the entire paper


from paddle.io import Dataset
import numpy as np
import os
import SimpleITK as sitk
import transforms as T
import re
import pandas as pd
import logging
from wodemodel import generate_model
import os
import paddle
print(paddle.__version__)

# Set the logging output level
logging.basicConfig(level=logging.INFO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
df = pd.read_excel('info.xlsx')
print(df.label)

SIZE = (20, 256, 256)

train_transforms = T.Compose([
    T.MRINormalize(),
    T.Resize3D(target_size=SIZE),  # Resampling
])

val_transforms = T.Compose([
    T.MRINormalize(),
    T.Resize3D(target_size=SIZE),
])


class MyDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.data_paths = []
        self.labels = []

        file_name = "train_data.txt" if mode == 'train' else "test_data.txt"
        with open(os.path.join(self.data_dir, file_name), "r", encoding="utf-8") as f:
            info = f.readlines()
        for img_info in info:
            img_info = img_info.rstrip()
            path, label = img_info.split(' ')  # Split string by space to get path and label
            print(path, label)

            self.data_paths.append(path)  # Add image path to self.data_paths
            self.labels.append(int(label))  # Convert label to integer and add to self.labels

    def __getitem__(self, index):
        img_path = self.data_paths[index]
        mask_path = re.sub(r'imgdata\\(\d+)_00.nii', r'imgdata\\\1_mask.nii', img_path)

        try:
            data = self.read_Nifit(img_path)
            mask = self.read_Nifit(mask_path)
        except ValueError as e:
            print(f"Error reading image or mask: {e}")
            return None, None

        try:
            data = self.maskcroppingbox(data, mask)
        except ValueError as e:
            print(f"Error applying mask: {e}")
            return None, None

        if self.transform is not None:
            transformed = self.transform(data)
            if isinstance(transformed, tuple):
                data, _ = transformed  # Assume transformation returns a tuple (data, label)
            else:
                data = transformed  # If only image data is returned

        if data.size == 0:
            logging.error(f"Transformed image size is zero, index: {index}")
            return None, None  # Return None if transformed image size is zero

        data = np.expand_dims(data, axis=0)
        label = self.labels[index]
        label = np.array([label], dtype="int32")

        return paddle.to_tensor(data), paddle.to_tensor(label)

    def read_Nifit(self, path):
        sitkImg = sitk.ReadImage(path)
        npImg = sitk.GetArrayFromImage(sitkImg)
        npImg = npImg.astype('float32')

        if npImg.size == 0:
            logging.error(f"Image {path} is empty or corrupted.")
            raise ValueError(f"Image {path} is empty or corrupted.")

        logging.info(f"Image {path} read successfully, size: {npImg.shape}")
        return npImg

    def maskcroppingbox(self, img, mask, use2D=False):
        if np.all(mask == 0):
            print("Mask is a zero array; cannot crop. Returning original image.")
            return img  # Return original image if mask is all zeros

        mask_2 = np.argwhere(mask)
        if mask_2.size == 0:
            print("Mask range is empty; cannot crop.")
            raise ValueError("Mask range is empty; cannot crop.")

        (zstart, ystart, xstart), (zstop, ystop, xstop) = mask_2.min(axis=0), mask_2.max(axis=0) + 1
        roi_image = img[zstart - 1:zstop + 1, ystart:ystop, xstart:xstop]
        return roi_image

    def __len__(self):
        return len(self.data_paths)

    def find_empty_transforms(self):
        empty_transforms_indices = []
        for i in range(len(self.data_paths)):
            try:
                img_path = self.data_paths[i]
                mask_path = re.sub(r'imgdata\\(\d+)_00.nii', r'imgdata\\\1_mask.nii', img_path)
                data = self.read_Nifit(img_path)
                mask = self.read_Nifit(mask_path)
                data = self.maskcroppingbox(data, mask)

                if self.transform is not None:
                    transformed = self.transform(data)
                    if isinstance(transformed, tuple):
                        data, _ = transformed
                    else:
                        data = transformed

                if data.size == 0:
                    empty_transforms_indices.append(img_path)

            except Exception as e:
                logging.error(f"Error occurred while processing index {i}: {e}")

        return empty_transforms_indices


train_dataset = MyDataset('', mode='train', transform=train_transforms)
val_dataset = MyDataset('', mode='val', transform=val_transforms)

layers = 50
load_layer_state_dict = paddle.load("MedicalNetparamemter/MedicalNet/resnet_" + str(layers) + ".pdiparams")
resnet3d = generate_model(layers, checkpoint=load_layer_state_dict)
model = paddle.Model(resnet3d)

# Use GPU for training
print("Model is training on:", paddle.get_device())

# Fine-tune with a smaller learning rate
opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
model.prepare(
    optimizer=opt,
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy()
)

save_dir = 'save_models' + str(layers)

class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self, target=0.5, path='./best_model', verbose=0):
        super().__init__()
        self.target = target
        self.epoch = None
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        if logs.get('acc') > self.target:
            self.target = logs.get('acc')
            self.model.save(self.path)
            print('Best accuracy is {} at epoch {}'.format(self.target, self.epoch))

save_dir = 'save_models' + str(layers)
callback_savebestmodel = SaveBestModel(target=0.5, path=save_dir + '/best_model')
model.fit(
    train_data=train_dataset,
    eval_data=val_dataset,
    batch_size=4,
    epochs=20,
    eval_freq=1,
    log_freq=20,
    save_dir=save_dir,
    save_freq=3,
    verbose=1,
    drop_last=False,
    shuffle=True,
    num_workers=1,
    callbacks=[callback_savebestmodel]
) 
