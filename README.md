# Spectral Densities of Power-Law and Natural Data

This repository houses code and figures for a blog post (coming soon) on my website about spectral densities of power-law data and natural data. 
Specifically:

- I use the Silverstein equation to numerically solve for the spectral density of structured Wishart matrices with power-law eigenvalues. 
- I show agreement with empirics
- I compare to natural datasets such as **CIFAR-10**, **MNIST**, and **Wikitext**
  
I find, extending observations made previously in the literature, that for CIFAR-10 the spectral density deviates substantially from a power law, but after ReLU mapping it is quite close with systematic deviations at small eigenvalues exhibiting log-oscillatory behavior.

![image](https://github.com/user-attachments/assets/2e7859ad-8eba-4c68-90ec-16cddc57f852)
![image](https://github.com/user-attachments/assets/389fcf9c-aeab-4bff-be26-bf3deb8f7652)



I also look at Wikitext with ReLU mapping, where choices about vocab_size given by custom BPE and about embedding dimension, must be made. The spectral density after ReLU again agrees quite well, with similar systematic deviations at small eigenvalues.

![image](https://github.com/user-attachments/assets/d33fa118-c017-4fb0-97cb-381a2419456e)
![image](https://github.com/user-attachments/assets/f92bdde2-6e6e-4c0e-a905-894798a987a7)

Blog post: [aaronjhf.github.io](https://aaronjhf.github.io/blog/power-law-spec/)
