# Spectral Densities of Power-Law and Natural Data

This repository houses code and figures for a blog post (coming soon) on my website about spectral densities of power-law data and natural data. 
Specifically:
-I use the Silverstein equation to numerically solve for the spectral density of structured Wishart matrices with power-law eigenvalues. 
-I show agreement with empirics
-I compare to natural datasets such as **CIFAR-10** and **MNIST**. 
We see, extending observations made previously in the literature, that for CIFAR-10 the spectral density deviates substantially from a power law, but after ReLU mapping it is quite close
with deviations at small eigenvalues.  For MNIST the difference is more dramatic, with the density seeming also near equiparameterization with a significant tail.

Planned blog post and figures: [aaronjhf.github.io](https://aaronjhf.github.io)
