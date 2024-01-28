import matplotlib.pyplot as plt

# Data
dino_values = [73.5, 75.25, 75.766, 76.046, 76.132]
dino_epochs = [100, 300, 400, 500, 600]
dino_hvs_epochs = [150, 300, 450]
# dino_hvs_epochs = [100, 200, 300]
dino_hvs_values = [74.67, 75.154, 76.56]

# Plotting
plt.plot(dino_epochs, dino_values, label='DINO (ViT-S/16)')
plt.plot(dino_hvs_epochs, dino_hvs_values, label='DINO (ViT-S/16) + our method')

# Labels and Title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ImageNet Linear Evaluation')

# TODO error bars: std

# Legend
plt.legend()

# Display the plot
# plt.savefig("pic.png")
plt.show()

