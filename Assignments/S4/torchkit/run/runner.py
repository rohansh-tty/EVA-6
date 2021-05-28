import torch
import torchvision
from tqdm import tqdm_notebook
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)

    train_loss, running_loss = [], 0
    correct = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # calculate nll loss
        loss = F.nll_loss(output, target)
        
        # Get the predictions and calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        s
        loss.backward()

        optimizer.step()
        running_loss += loss
        pbar.set_description(desc= f'Train set: batch_id={batch_idx}  Average loss: {loss} Accuracy: {round(100*correct/len(train_loader.dataset),3)}')

    _loss =  running_loss/len(train_loader.dataset)
    _correct = correct/len(train_loader.dataset)
    train_loss.append(_loss)
    # correct.append(_correct)
    tb.add_scalar('Train Loss', _loss, epoch)
    tb.add_scalar('Train Accuracy', _correct, epoch)
    tb.close()
    


def test(model, device, test_loader, epoch):
    model.eval()
    test_losses, test_loss = [], 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')#.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    _correct = correct/len(test_loader.dataset)
    test_losses.append(test_loss)
    # correct.append(_correct)

    tb.add_scalar('Test Loss', test_loss, epoch)
    tb.add_scalar('Test Accuracy', _correct, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * _correct))
    
    
    tb.close()