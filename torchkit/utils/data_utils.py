import matplotlib.pyplot as plt
import numpy as np

def plot_images(data,file):
  r = len(data)//5
  # fig, axes = plt.subplots(12, 5, figsize=(25, 25)) # this means that create a figure of 4 rows with 5 columns and figsize is 15x15
  fig, axes = plt.subplots(r, 5, figsize=(15, 15))
  
  # Plot the images
  for i, img in enumerate(data):
      # Get the image
      image, label, pred = img['image'], img['label'].cpu().numpy()[0], img['pred'].cpu().numpy()[0]
      img = image.cpu().numpy().astype(np.uint8).reshape(28,28) # convert the image tensor to np array of shape 28x28, 2d image
      
      # Get the appropriate subplot
      x  = i%5         # Subplot x-coordinate
      y  = int(i/5)    # Subplot y-coordinate
      ax = axes[y][x]
      ax.imshow(img, cmap='gray')

      # Format the plot
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      ax.set_title(f'Label: {label} Prediction: {pred}')

   # save the plot
  plt.savefig(file+'.png')
#   files.download(file+'.png')
      


