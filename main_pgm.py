# This part is for training the PGM model
# It includes data preprocessing, model initialization, and the training loop
# It save best_model

# Create DataSet and Transform to read medical data, convert to numpy, normalize, and resample to appropriate size
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
# Set log output level
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

        logging.info(f"Image {path} read successfully, shape: {npImg.shape}")
        return npImg

    def maskcroppingbox(self, img, mask, use2D=False):
        if np.all(mask == 0):
            print("Mask is all-zero array, cannot crop. Returning original image.")
            return img  # Return original image if mask is all zero

        mask_2 = np.argwhere(mask)
        if mask_2.size == 0:
            print("Mask range is empty, cannot crop.")
            raise ValueError("Mask range is empty, cannot crop.")

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
                logging.error(f"Error processing index {i}: {e}")

        return empty_transforms_indices


train_dataset = MyDataset('', mode='train', transform=train_transforms)
val_dataset = MyDataset('', mode='val', transform=val_transforms)

# Initialize your model and optimizer, etc.
layers = 50
load_layer_state_dict = paddle.load("MedicalNetparamemter/MedicalNet/resnet_" + str(layers) + ".pdiparams")
resnet3d = generate_model(layers, checkpoint=load_layer_state_dict)
model = paddle.Model(resnet3d)

# Use a smaller learning rate for fine-tuning
opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

# Assume train_dataset and val_dataset are already prepared
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=23, shuffle=False, num_workers=1, drop_last=False)

# Instantiate TPGM object
norm_mode = "l2"  # or "mars" depending on your needs
tpgm = TPGM(resnet3d, norm_mode=norm_mode, exclude_list=[])
tpgm_trainer_instance = tpgm_trainer(model=resnet3d, pgmloader=train_loader, norm_mode=norm_mode, proj_lr=0.0001,
                                     max_iters=10)


class SaveBestModel:
    def __init__(self, model, target=0.5, path='./best_model', verbose=0):
        self.model = model  # Directly pass the model to be operated on
        self.target = target
        self.best_acc = target  # Used to compare and update the best accuracy
        self.path = path
        self.verbose = verbose

    def save_if_best(self, acc, epoch):
        if acc > self.best_acc:
            self.best_acc = acc
            # Use paddle.save to save model parameters
            paddle.save(self.model.state_dict(), self.path)
            if self.verbose:
                print(f'Best model saved with accuracy: {self.best_acc} at epoch {epoch}')


save_best_model = SaveBestModel(resnet3d, target=0.5, path='save_models' + str(layers) + '/best_model2', verbose=1)

for epoch in range(20):
    resnet3d.train()  # Use the model's .train() method
    for batch_id, data in enumerate(train_loader):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)

        # Apply TPGM constraint and update model
        tpgm_trainer_instance.tpgm_iters(resnet3d, apply=False)

        # Normal training process
        preds = resnet3d(images)  # Use the underlying network object for forward propagation
        loss = loss_fn(preds, labels)
        loss.backward()
        opt.step()
        opt.clear_grad()

    # Validation cycle
    resnet3d.eval()  # Use the model's .eval() method
    accs = []
    for batch_id, data in enumerate(val_loader):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        preds = resnet3d(images)
        acc = metric.compute(preds, labels)
        print(f'acc{acc}')
        # Ensure acc is a scalar
        acc_scalar = acc.numpy()
        accs.append(acc_scalar)
        print(f'accs{accs}')

    # Now accs contains scalars, can safely calculate average
    epoch_acc = sum(accs) / len(accs)
    print(f'Epoch {epoch}, Validation Accuracy: {epoch_acc}')
    save_best_model.save_if_best(epoch_acc, epoch)

    metric.reset()

tpgm_trainer_instance.tpgm_iters(resnet3d, apply=True)
