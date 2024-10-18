# CS 5787 Deep Learning Assignment #3 - VAE & GAN

Authors: Mitchell Krieger

This repository contains materials to complete the [third assigment](./CS%205787%20-%20EX%203.pdf) for CS 5787 at Cornell Tech in the Fall 2024 semester. There are two parts to this assignment, the theoretical part and the practical part. 

## Theoretical

The first part is theoretical. All of its materials including a LaTex file and its PDF output can be found in the [report](./report/) directory.

## Practical

The second part of the assignment is practical. It is contained in the [assignment2_practical.ipynb](./assignment2_practical.ipynb) notebook. 

### Setup
All code, including data download, model definitions, training loops, hyperparameter tuning and evaluation should be runable from top to bottom of the notebook with reproduceable results by creating a virtual environment, installing the packages in requirements.txt, and logging into Weights & Biases.

```bash
source -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
wandb login
```

If you do not login to Weights & Biases via the command line the first cell of the notebook will attempt to login again. If you do not log in, you must set `mode` argument the `wandb.init()` method to `offline`. 

If cuda or mps is available to train models on GPUs, the notebook will attempt to use cuda, otherwise it will default to using cpu. If you wish to use mps, uncomment that section in `Check for GPU Access`.

### Data

For this assignment we are using the Fashion MNIST dataset. Running the first cell will download the data using `wget` to the `data` directory. After that the `FashionMNISTDataset` pytorch Dataset will handle loading the data into tensors and can retrieve data like any pytorch dataset can. In addition, a utility function `show_img` is provided to display any given sample given its index. 

```python
gen = torch.Generator().manual_seed(123)

train = FasionMNISTDataset(PATH, 'train', device=device)
```

### Variational Autoencoder with SVM

#### Models & Training

First we attempt to create a generative model of the FashionMNIST Dataset using a Variational Autoencoder (VAE) with an SVM Head using a semi-supervised approach. The VAE using the Gaussian reparameterization trick that returns a sample from the gaussian parameterized by `mu` and `log_var`. We feed an equal amount of labeled and unlabeled data to the model to generate samples. We use this trick to train an encoder that uses this trick to generate a latent representation of the image and a decoder that generates an image from the latent representation. An SVM Classifier is trained as a head on the final latent representations. 

VAE with SVM modeling can be found in the [vae_svm.ipynb](./vae_svm.ipynb) notebook. In the notebook, there are 4 main pytorch modules:
- `Gaussian`: Responsible for the reparameterization trick
- `Encoder`: Generates latent vector representations of images
- `Decoder`: Generates images from latent vector representations
- `VAE`: Module to organizing the training of all above modules

Lastly sci-kit learn's `SVC` module with a radial basis function kernel is trained on the output of `VAE`.

#### Loading, Saving & Testing

Trained weights for the VAE have been saved to the `models/vae` directory as `.pth` files using the simple `torch.save()` method. A binary pickle file of scikit-learn `SVC` models can be found in the same directory. Weights and models can be loaded for generation and testing by the following:

```python
# Load VAE model weights
tmp = VAE(input, hidden, latent).to(device)
tmp.load_state_dict(torch.load('/content/drive/MyDrive/DL3/vae_1000.pth'))

# Load SVM Head from picklefile
with open('/content/drive/MyDrive/DL3/svm_1000.pkl', 'rb') as f:
    head = pickle.load(f)

def generate_images(model, head, num_images, latent_dim, device):
    # Set the model to evaluation mode
    model.eval()

    # Sample from a standard normal distribution in the latent space
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)  # latent_dim is the size of the latent vector
        generated_images = model.decoder(z)  # Pass through decoder to get reconstructed images
        generated_images = generated_images.view(num_images, 28, 28).cpu()  # Reshape and move to CPU

    preds = head.predict(z.cpu())

    labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }

    # Plot the generated images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i in range(num_images):
        axes[i].imshow(generated_images[i], cmap='gray')
        axes[i].set_title(labels_map[preds[i]])
        axes[i].axis('off')
    plt.show()

# Example usage:
num_images = 10  # Specify how many images you want to generate
latent_dim = 10  # Latent dimension used during training
generate_images(vae, head, num_images, latent_dim, device)
```



### DCGAN, WGAN, WGAN-GP

#### Models & Training

Next we attemped 6 different GAN models to generate FashionMNIST images. There were three methods used DCGAN, WGAN, and WGAN-GP. Each method was attempted with two architectures A & B. All code for these 6 models can be found in [gan.ipynb](./gan.ipynb) These architectures are stored in two pytorch modules:
- `Generator`: Generates new fake images
- `Discriminator`: Discriminates or (Critiques or WGAN and WGAN-GP) between real and fake images

In both of these modules, the architecture type A & B can be selected by initializing a `Generator` or `Discriminator` class and passing `'a'` or `'b'`. Alternatively, wrapper classes to collect all training and generation elements of DCGAN, WGAN, and WGAN-GP are provided. `'a'` or `'b'` can also be passed as arguments there as well. A train method is provided in each one of these classes:

```
# Constructor args: architechture_type, z_dimension, num_classes, channels, feature_multiple, device
dc = DCGAN('a', 100, 10, 1, 64, device)

# Train args: dataloader, num_epochs, batch_size, z_dimension
dc.train(traindataloader, 10, 64, 100)
```

#### Loading, Saving & Testing

Trained weights have been saved to the `models` directory as `.pth` files using the simple `torch.save()` method. Weights can be loaded for generation and testing by the following:

```python
m = Generator('dc', 'a', 100, 10, 1, 64)
m.load_state_dict(torch.load('/content/drive/MyDrive/DL3/dc_generator_arch_a.pth'))
```

Models can be tested after loading using:
```
labels_map = {
             "T-Shirt": 0,
             "Trouser": 1,
             "Pullover": 2,
             "Dress": 3,
             "Coat": 4,
             "Sandal": 5,
             "Shirt": 6,
             "Sneaker": 7,
             "Bag": 8,
             "Ankle Boot": 9,
        }

def generate_images(model, num_images, labels):
    model.generator.eval()
    with torch.no_grad():  
        noise = torch.randn(num_images, model.z_dim, 1, 1, device=model.device)
        generated_images = model.generator(noise, labels)
        # generated_images = (generated_images + 1) / 2
        grid = torchvision.utils.make_grid(generated_images.cpu(), nrow=4, padding=2, normalize=True)
        np_grid = grid.permute(1, 2, 0).numpy()
        plt.imshow(np_grid)
        plt.axis('off')

num_imgs = 4
labels = [labels_map["Dress"]] * num_imgs
generate_images(dc, num_imgs, F.one_hot(torch.tensor(labels, device=device), num_classes=10).float())
```

Discriminators/Critics can also be loaded using the same methods.