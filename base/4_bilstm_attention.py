import time
from sklearn.externals import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 512
N_EPOCHS = 3


x_train, x_test, y_train, features, test_features, word_index = load_and_prec()
train_idx = joblib.load('train_idx.pkl')
valid_idx = joblib.load('valid_idx.pkl')


train_preds = np.zeros((len(x_train)))
test_preds = np.zeros((len(x_test)))

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

kfold_X_features = features[train_idx.astype(int)]
kfold_X_valid_features = features[valid_idx.astype(int)]
x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

model = NeuralNetBiLSTM()
model.cuda()
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
step_size = 300
base_lr, max_lr = 0.001, 0.003
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=max_lr)

scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                     step_size=step_size, mode='exp_range',
                     gamma=0.99994)

train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

train = MyDataset(train)
valid = MyDataset(valid)

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(N_EPOCHS):
    start_time = time.time()
    model.train()

    avg_loss = 0.
    for i, (x_batch, y_batch, index) in enumerate(train_loader):
        f = kfold_X_features[index]
        y_pred = model([x_batch,f])
        if scheduler:
            scheduler.batch_step()
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    model.eval()
    valid_preds_fold = np.zeros((len(valid_idx)))
    test_preds_fold = np.zeros((len(x_test)))
    avg_val_loss = 0.
    for i, (x_batch, y_batch, index) in enumerate(valid_loader):
        f = kfold_X_valid_features[index]
        y_pred = model([x_batch,f]).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds_fold[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]
    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, N_EPOCHS, avg_loss, avg_val_loss, elapsed_time))
avg_losses_f.append(avg_loss)
avg_val_losses_f.append(avg_val_loss)
# predict all samples in the test set batch per batch
for i, (x_batch,) in enumerate(test_loader):
    f = test_features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    y_pred = model([x_batch,f]).detach()
    test_preds_fold[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]
train_preds[valid_idx] = valid_preds_fold
test_preds += test_preds_fold / 1

joblib.dump(valid_preds_fold, 'valid_pred_bilstm.pkl', compress=3)
joblib.dump(test_preds, 'test_pred_bilstm.pkl', compress=3)

print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))
print(valid_preds_fold.shape)
delta = bestThresshold(y_train[valid_idx.astype(int)],valid_preds_fold)

