# Data-Science-Machine-Learning-Hyperparameter-Tuning
Introduction
This repository contains an analysis of the MNIST dataset and the development of various machine learning models for digit recognition tasks. The MNIST dataset is a widely used benchmark dataset in the machine learning community, consisting of grayscale images of handwritten digits (0 to 9) and corresponding labels. The goal of this project is to explore the dataset, preprocess the data, build classification models to predict the digit labels, and deploy the best-performing model using Streamlit for web-based inference.


Project Structure

HPO.ipynb: Jupyter Notebook containing hyperparameter optimization for the CNN model using Optuna.
mnist_model.h5: Pre-trained CNN model file.
mnist_streamlit.py: Streamlit deployment file for web-based inference.
mnist_streamlit_predict.py: Streamlit deployment file for making predictions.
sample_image.webp: Sample image for testing the deployed model.
README.md: Readme file providing an overview of the project, its objectives, and instructions for running the code.
requirments.txt: File containing the required libraries for running the code.


Dataset

The MNIST dataset is loaded using the datasets.load_digits() function from the sklearn library.
It consists of 1,797 samples, where each sample is an 8x8 array of pixel values representing a handwritten digit.
The dataset is divided into features (pixel values) and labels (digit labels).


Analysis and Model Building

The analysis begins with data visualization, where the first 10 samples from the dataset are displayed as grayscale images.
Data preprocessing techniques such as splitting the dataset into training and testing sets and scaling the features are applied.
Several classification models are trained and evaluated on the dataset, including Random Forest, SVM, KNN, Naive Bayes, Decision Tree, Gradient Boosting, CNN, RNN, and ANN.
Model evaluation is performed using cross-validation with 3 folds to assess each model's accuracy.
Hyperparameter tuning techniques such as grid search, random search, and Bayesian optimization are employed to optimize the performance of the CNN model.


Deployment with Streamlit

The best-performing model (CNN) is deployed for web-based inference using Streamlit.
The mnist_streamlit.py file contains the code for deploying the model.
Users can upload handwritten digit images, and the deployed model predicts the digit label.
The mnist_streamlit_predict.py file is used for making predictions with the deployed model.
The Streamlit app provides an intuitive interface for users to interact with the model and visualize the prediction results.


Instructions for Running the Code

Clone the repository to your local machine.
Ensure you have Python installed along with necessary libraries (scikit-learn, pandas, numpy, seaborn, matplotlib, keras, optuna, streamlit).
Install the required libraries listed in requirements.txt.
Open and run the Jupyter Notebook HPO.ipynb for hyperparameter optimization.
Use the pre-trained model file mnist_model.h5 for deployment with Streamlit.
Run the Streamlit app using the command streamlit run mnist_streamlit.py and access it through the provided URL.
Use sample_image.webp for testing the deployed model.


Contributing

Contributions to the project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


License

This project is licensed under the MIT License. Feel free to use and modify the code for your own purposes.
