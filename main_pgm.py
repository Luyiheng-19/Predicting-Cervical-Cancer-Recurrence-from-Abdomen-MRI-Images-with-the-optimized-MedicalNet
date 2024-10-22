from wodemodel import generate_model
import os
import paddle
from paddle.io import DataLoader
from tpgm import TPGM, tpgm_trainer  # Make sure to import your TPGM implementation

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

# Apply TPGM constraint
tpgm_trainer_instance.tpgm_iters(resnet3d, apply=True)
