# GoodReads
This project will build a recommendation system for GoodReads. The dataset is available on https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

Currently, we have done the following:

1. Use a singular value decomposition based model called matrix factorization (https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) to create a baseline model. We use keras functional api to implement gradient descent, and our model includes bias and regularization terms for both book and users.

2. We use word2vec (skip-gram) schema to build user and book embeddings (100 dimension for each) in keras. The book vector shows high correlation with reality: for example, the most similiar book of a japanese manga are also japense manga.

3. We use a deep and wide neural network architecture (https://arxiv.org/abs/1606.07792) which includes interaction between user and book to predict the rating score.

4. Our proposed model improve the mse from 3.5 to 1.2.
