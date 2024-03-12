# Generative-AI
A [Generative Adversarial Network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network) consists of two neural networks, the **generator** and the **discriminator**, which are trained simultaneously through a competitive process.

![image](https://github.com/OmarAzizi/Generative-AI/assets/110500643/135093fe-0be1-43f6-bf15-63ed145b3fec)


1. Generator: Its goal is to create data that is indistinguishable from real data.
2. Discriminator: The discriminator network tries to distinguish between real data and data generated by the generator. It is trained on both real data and the fake data generated by the generator.

## About This Model
I used the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset to train the GAN for generating hand-written digits. Then I used [matplotlib](https://matplotlib.org/) to plot some generated images on each epoch.
```bash
  | Name          | Type          | Params
------------------------------------------------
0 | generator     | Generator     | 358 K 
1 | discriminator | Discriminator | 21.4 K
-----------------------------------------------
```


### Before Training
The images generated were basically just noise

![image](https://github.com/OmarAzizi/Generative-AI/assets/110500643/5de7306a-a8d8-49c4-8070-a1dfd842970e)

### After First Epoch

![image](https://github.com/OmarAzizi/Generative-AI/assets/110500643/2e467cfb-9e01-4011-837f-cddcdbbcb169)

### After 10 Epochs
Generated images pattern started to look like a hand-written digit 

![image](https://github.com/OmarAzizi/Generative-AI/assets/110500643/f5f2393b-64d6-4acd-9011-12997ec61ced)


### After 100 Epochs
Generated images now look like hand-written digits now

![image](https://github.com/OmarAzizi/Generative-AI/assets/110500643/4aa36be2-afd9-4a39-841b-d61b11cf5902)

