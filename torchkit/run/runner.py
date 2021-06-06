import torch
import numpy as np
import torchvision
import torch.optim as optim
from tqdm import tqdm_notebook
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

misclassified_images = []


tb = SummaryWriter()


def train(model, epoch, config=None):
    model.train()
    pbar = tqdm(config.trainloader)
    train_misc_images = []

    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer=optimizer, **config.lr_scheduler_params[config.lr_scheduler])
    train_loss, running_loss = [], 0
    correct = 0
    count = 0
   
    for batch_idx, (data, target) in enumerate(pbar):
        count += 1
        data, target = data.to(config.device), target.to(config.device)
        optimizer.zero_grad()
        output = model(data)
        lr_array = []

        # calculate nll loss
        loss = F.nll_loss(output, target)
        
        # Get the predictions and calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()
        
        if config.lr_scheduler=='OneCycleLR':
            scheduler.step()
            lr_array.append(scheduler.get_last_lr())
            
        running_loss += loss

         # Misclassified Images
        if config.misclassified:
          result = pred.eq(target.view_as(pred))
          if count > 430 and count < 460:
              for i in range(0, config.trainloader.batch_size):
                  if not result[i]:
                      train_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': data[i]})
        pbar.set_description(desc= f'Train set: batch_id={batch_idx}  Average loss: {loss} Accuracy: {round(100*correct/len(config.trainloader.dataset),3)}')
    
    if not config.lr_scheduler=='ReduceLROnPlateau':
        lr = np.array(scheduler.get_last_lr())
        tb.add_scalar('LR', lr, epoch)
    
    
    
    train_loss_value = running_loss/len(config.trainloader.dataset)
    train_acc_value = correct/len(config.trainloader.dataset)
    
    
    
    tb.add_scalar('Train Loss', train_loss_value, epoch)
    tb.add_scalar('Train Accuracy', train_acc_value, epoch)
    
    return train_misc_images

        



def test(model, epoch, config=None):
    model.eval()
    test_losses, test_loss = [], 0
    correct = 0
    count = 0
    test_misc_images = []
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer=optimizer, **config.lr_scheduler_params[config.lr_scheduler])
    img, label = next(iter(config.testloader))
    test_input = img.to(config.device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(config.testloader):
            count += 1
            
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')#.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            result = pred.eq(target.view_as(pred))
            
            if config.misclassified:
              if count >40  and count < 70:
                  for i in range(0, config.testloader.batch_size):
                      if not result[i]:
                          test_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': data[i]})
   
    
    test_loss_value = test_loss/len(config.testloader.dataset)
    test_acc_value = correct/len(config.testloader.dataset)
    test_losses.append(test_loss)
   

    
    tb.add_scalar('Test Loss', test_loss_value, epoch)
    tb.add_scalar('Test Accuracy', test_acc_value, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss/len(config.testloader.dataset), correct, len(config.testloader.dataset),
        100. * test_acc_value))
    print('stepping scheduler..')
    # scheduler.step()

    tb.add_graph(model,test_input )

    # tb.close()
    
    return test_misc_images, test_loss
    
    
    
def fit(model, config):
    # model = model().to(config.device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    # misc=misclassified
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer=optimizer, **config.lr_scheduler_params[config.lr_scheduler])
    
    for epoch in range(1,config.EPOCHS+1):
        print('\nEPOCH: ', epoch,'LR: ',scheduler.get_last_lr())
      
        # Train and Test Model
        train_misc_img = train(model, epoch, config=config)
        # if config.lr_scheduler=='OneCycleLR':
        #     scheduler.step()
        test_misc_img, test_loss = test(model, epoch, config=config)
        
        # Scheduler Step, update Learning Rate
        if config.lr_scheduler=='ReduceLROnPlateau':
            scheduler.step(test_loss)
        if config.lr_scheduler=='StepLR':
            scheduler.step()
            
         # add lr to tensorboard
        lr = np.array(scheduler.get_last_lr())
        tb.add_scalar('Learning Rate', lr, epoch)
    tb.close()
        

    return train_misc_img, test_misc_img
