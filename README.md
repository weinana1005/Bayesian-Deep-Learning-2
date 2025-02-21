# COMP0171-Bayesian-Deep-Learning-CW2

- **Final Grades:** 93/100  
  - **Comments:**
    - [PART 1 (17/19)] Good implementation; For free response, partial credit; yes inference methods are different, but in reason #2 why does the different posterior estimate have such an effect in data space?
    - [PART 2 (11/11)] For part 2, what does temperature scaling do when the data near the decision boundary looks like in the paper? Works well, but fully connected layers have lots of parameters (so no EC)
---

## Overview

This repository contains coursework solutions for **COMP0171: Bayesian Deep Learning**. It explores Bayesian methods in deep learning, emphasizing probabilistic modeling, uncertainty quantification, and posterior inference.

Key topics covered:
- **Probabilistic Modeling**: Understanding Bayesian approaches for parameter estimation.
- **Uncertainty Quantification**: Bayesian classifiers and variational inference.
- **Posterior Estimation**: Approximating distributions of parameters given data.

The coursework consists of two parts:
1. **Uncertainty Quantification** (Bayesian inference, MCMC, Laplace Approximation)
2. **Variational Autoencoders (VAEs)** (Probabilistic deep generative models)

Each section contains a Jupyter Notebook with theoretical explanations, implementations, and experimental results.

---

## Project Structure

### **(Part 1) Uncertainty Quantification.ipynb**
*Description:* This notebook explores Bayesian inference in deep learning, covering:
- **Bayesian parameter estimation**
- **Markov Chain Monte Carlo (MCMC) sampling**
- **Laplace approximation for uncertainty estimation**
- **Posterior analysis and visualization**

### **(Part 2) Variational Auto-Encoder.ipynb**
*Description:* This notebook implements a Variational Autoencoder (VAE) for generative modeling, covering:
- **Latent variable models and approximate inference**
- **Evidence Lower Bound (ELBO) and KL divergence regularization**
- **Reparameterization trick for efficient backpropagation**
- **Training and evaluating VAEs on real-world datasets**

---

## Project Structure

### **Part 1: Uncertainty Quantification**
- **MCMC Sampling:** Implemented Metropolis-Hastings algorithm for Bayesian parameter estimation.
- **Laplace Approximation:** Used Hessian-based approximation for posterior estimation.
- **Model Evidence:** Compared Bayesian feature mappings using log marginal likelihood.

### **Part 2: Variational Autoencoders (VAEs)**
- **Probabilistic Latent Representations:** Trained a generative model using a learned latent space.
- **Reparameterization Trick:** Enabled backpropagation through stochastic latent variables.
- **KL Regularization:** Balanced latent variable regularization with reconstruction loss.

---
#### **Part 1 Key Implementations**

**1. Dataset Loading & Visualization**

```python

dataset, validation_set = torch.load("two_moons.pt")
x_train, y_train = dataset.tensors
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='autumn', edgecolor='k')
plt.xlim(-3,3)
plt.ylim(-2,2)
plt.show()
```

- Loads a toy **Two Moons** dataset, a common benchmark for classification tasks.
- Visualizes the dataset, where two distinct classes are color-coded.

**2. Simple Feedforward Network**

```python
import torch.nn as nn

class TwoMoonsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        h = self.net(x)
        return torch.sigmoid(h).squeeze(1)

network = TwoMoonsNetwork()
```

- Defines a **simple feedforward neural network** for binary classification.
- Uses **ReLU activations** and a **sigmoid** at the output for probability estimation.

---
## Part 1 Task 1
This implementation defines the **log-likelihood** of the data under a **Bernoulli distribution** and the prior for the Bayesian model:

```python
def log_likelihood(network, X, y):
    """
    Computes the log probability `log p(y | x, theta)` for a batch of inputs X.
    
    INPUT:
    - network: instance of classifier network, extends `nn.Module`
    - X: batch of inputs; `torch.FloatTensor`, matrix of shape = `(batch_size, 2)`
    - y: batch of targets; `torch.FloatTensor`, vector of shape = `(batch_size,)`
    
    OUTPUT:
    - lp: log probability value of `log p(y | x, theta)`; scalar
    """
    
    # Forward pass through the network to obtain predictions
    y_pred = network(X)
    
    # Define the Bernoulli distribution for the likelihood
    bernoulli_dist = dist.Bernoulli(probs=y_pred)
    
    # Compute the log probability of the true labels under the predicted distribution
    lp = bernoulli_dist.log_prob(y)
    
    # Sum over the batch to obtain a scalar value
    return lp.sum()
```

### **Explanation**
- **`network(X)`**: Passes the input `X` through the neural network to generate predicted probabilities.
- **Bernoulli Distribution**: Since the output is a probability, we assume a Bernoulli distribution for the likelihood.
- **`bernoulli_dist.log_prob(y)`**: Computes the log probability of observing the actual labels `y` under the predicted probabilities.
- **Summation**: The log-likelihood values across the batch are summed to return a scalar value.

This function plays a crucial role in Bayesian inference, allowing for **posterior estimation** using methods like Markov Chain Monte Carlo (MCMC) or Stochastic Gradient Langevin Dynamics (SGLD).

```python
def log_prior(network):
    """
    Computes the log prior probability `log p(theta)` assuming a standard normal distribution.
    
    INPUT:
    - network: instance of classifier network, extends `nn.Module`
    
    OUTPUT:
    - log_p: log probability value of `log p(theta)`; scalar
    """
    
    log_p = 0.0
    for param in network.parameters():
        normal_dist = dist.Normal(0.0, 1.0)  # Define the normal distribution with mean 0 and variance 1
        log_p += normal_dist.log_prob(param).sum()
    
    return log_p
```

### **Explanation**
- **`log_prior(network)`**: Computes the log prior probability assuming a **Gaussian prior** with mean `0` and variance `1`.
- **Iteration over `network.parameters()`**: Accesses all parameters of the network.
- **Standard Normal Distribution**: The prior distribution is modeled as `𝒩(0,1)`.
- **Summation**: Aggregates the log probabilities over all parameters to return a scalar value.

This function is essential for Bayesian inference, ensuring **regularization** and preventing overfitting by enforcing prior constraints on the network parameters.

(UNDONE)
