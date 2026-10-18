
import os
import ast
import warnings
import numpy as np
import pandas as pd
import random
import joblib
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
import tensorflow.keras.backend as K
from functools import partial
from tensorflow.python.framework.ops import disable_eager_execution

import numpy as np
import joblib
import sklearn
from joblib.numpy_pickle import NumpyUnpickler

disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

# Load real-world data variables: mean_train, cov_train, mean_test, cov_test, f_hat
rw_data = np.load('data/rw_data.npz')
mean_train = rw_data['mean_train']
cov_train = rw_data['cov_train']
mean_test = rw_data['mean_test']
cov_test = rw_data['cov_test']
diff_vec = mean_test - mean_train
sig_y = rw_data['sig_y']

print("Loaded keys:", list(rw_data.files))
print("mean_train shape:", mean_train.shape)
print("cov_train shape:", cov_train.shape)
print("mean_test shape:", mean_test.shape)
print("cov_test shape:", cov_test.shape)
print("sig_y:", float(np.asarray(sig_y).squeeze()))

nonlin = 'rf'
if nonlin == 'rf':
    f_hat = joblib.load("data/rf_model.pickle")
elif nonlin == 'gbm':
    f_hat = lgb.Booster(model_file='data/lgb_model.txt')

print(rw_data)

N = 5830
gauss_mv_samples = np.random.multivariate_normal(mean_train, cov_train, size=N)

x_1 = gauss_mv_samples[:,0:1]
x_2 = gauss_mv_samples[:,1:2]
x_3 = gauss_mv_samples[:,2:3]
x_4 = gauss_mv_samples[:,3:4]
x_5 = gauss_mv_samples[:,4:5]
x_6 = gauss_mv_samples[:,5:6]
x_7 = gauss_mv_samples[:,6:7]
x_8 = gauss_mv_samples[:,7:8]
x_9 = gauss_mv_samples[:,8:9]
x_10 = gauss_mv_samples[:,9:10]
x_11 = gauss_mv_samples[:,10:11]
x_12 = gauss_mv_samples[:,11:12]
x_13 = gauss_mv_samples[:,12:13]
x_14 = gauss_mv_samples[:,13:14]
x_15 = gauss_mv_samples[:,14:15]
x_16 = gauss_mv_samples[:,15:16]
x_17 = gauss_mv_samples[:,16:17]
x_18 = gauss_mv_samples[:,17:18]
x_19 = gauss_mv_samples[:,18:19]
x_20 = gauss_mv_samples[:,19:20]
X_train = np.concatenate([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20], axis = 1)

print("X_train sample:\n", pd.DataFrame(X_train, columns=[f"Feature_{i}" for i in range(1, 21)]).head())

error = 'gaussian'
if error == 'gaussian':
    y_train = f_hat.predict(X_train).reshape((N,1)) + np.random.normal(loc = 0, scale = sig_y, size = (N,1))
elif error == 'uniform':
    y_train = f_hat.predict(X_train).reshape((N,1)) + np.random.uniform(low = -(np.sqrt(12)*sig_y)/2, high = (np.sqrt(12)*sig_y)/2, size = (N,1))
    



test_size = 3866
thetas = [4]
for theta in thetas:
    gauss_mv_samples = np.random.multivariate_normal(mean_train + theta*diff_vec, cov_test, size=test_size)
    x_1 = gauss_mv_samples[:,0:1]
    x_2 = gauss_mv_samples[:,1:2]
    x_3 = gauss_mv_samples[:,2:3]
    x_4 = gauss_mv_samples[:,3:4]
    x_5 = gauss_mv_samples[:,4:5]
    x_6 = gauss_mv_samples[:,5:6]
    x_7 = gauss_mv_samples[:,6:7]
    x_8 = gauss_mv_samples[:,7:8]
    x_9 = gauss_mv_samples[:,8:9]
    x_10 = gauss_mv_samples[:,9:10]
    x_11 = gauss_mv_samples[:,10:11]
    x_12 = gauss_mv_samples[:,11:12]
    x_13 = gauss_mv_samples[:,12:13]
    x_14 = gauss_mv_samples[:,13:14]
    x_15 = gauss_mv_samples[:,14:15]
    x_16 = gauss_mv_samples[:,15:16]
    x_17 = gauss_mv_samples[:,16:17]
    x_18 = gauss_mv_samples[:,17:18]
    x_19 = gauss_mv_samples[:,18:19]
    x_20 = gauss_mv_samples[:,19:20]
    X_test = np.concatenate([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20], axis = 1)

    error = 'gaussian'
    if error == 'gaussian':
        y_test = f_hat.predict(X_test).reshape((test_size,1)) + np.random.normal(loc = 0, scale = sig_y, size = (test_size,1))
    elif error == 'uniform':
        y_test = f_hat.predict(X_test).reshape((test_size,1)) + np.random.uniform(low = -(np.sqrt(12)*sig_y)/2, high = (np.sqrt(12)*sig_y)/2, size = (test_size,1))

    print(f"Source feature matrix: {X_train.shape}")
    print(f"Target feature matrix (theta={max(thetas)}): {X_test.shape}")

    import matplotlib.pyplot as plt
    import seaborn as sns    
    
    # Set the plotting style
    sns.set_theme(style="whitegrid")

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot target distributions (X_train vs X_test)
    sns.histplot(X_train.flatten(), kde=True, color='blue', alpha=0.5, label='X_train', ax=axes[0])
    sns.histplot(X_test.flatten(), kde=True, color='orange', alpha=0.5, label='X_test', ax=axes[0])
    axes[0].set_title('Feature Distributions (X_train vs X_test)')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot feature distributions (average values across samples for each of the 20 features)
    mean_features_train = X_train.mean(axis=0)
    mean_features_test = X_test.mean(axis=0)

    axes[1].plot(range(1, 21), mean_features_train, marker='o', linestyle='-', color='blue', label='X_train Mean')
    axes[1].plot(range(1, 21), mean_features_test, marker='s', linestyle='--', color='orange', label='X_test Mean')
    axes[1].set_title('Feature Means (X_train vs X_test)')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Mean Value')
    axes[1].set_xticks(range(1, 21))
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a DataFrame for X_test for easier correlation calculation and labeling
    feature_names = [f'Feature_{i}' for i in range(1, 21)]
    df_test = pd.DataFrame(X_test, columns=feature_names)

    # Calculate the Pearson correlation matrix
    corr_matrix = df_test.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5
    )
    plt.title('Correlation Matrix of X_test Features', fontsize=16)
    plt.tight_layout()
    plt.show()

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create DataFrames for both datasets
    feature_names = [f'Feature_{i}' for i in range(1, 21)]
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_test = pd.DataFrame(X_test, columns=feature_names)

    # Compute correlation matrices
    corr_train = df_train.corr()
    corr_test = df_test.corr()

    # Plot the correlation matrices side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Train correlation heatmap
    sns.heatmap(
        corr_train,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        cbar=True,
        ax=axes[0],
        linewidths=0.5
    )
    axes[0].set_title('Correlation Matrix of X_train Features', fontsize=16)

    # Test correlation heatmap
    sns.heatmap(
        corr_test,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        cbar=True,
        ax=axes[1],
        linewidths=0.5
    )
    axes[1].set_title('Correlation Matrix of X_test Features', fontsize=16)

    plt.tight_layout()
    plt.show()


    from scipy.stats import wasserstein_distance
    import numpy as np

    print(f"{'Feature':<12} | {'Wasserstein Distance':<20}")
    print("-" * 38)

    feature_w_distances = []
    for i in range(20):
        dist = wasserstein_distance(X_train[:, i], X_test[:, i])
        feature_w_distances.append(dist)
        print(f"Feature_{i+1:<5} | {dist:<20.4f}")

    # Compute the average univariate Wasserstein distance as a summary metric
    avg_w_dist = np.mean(feature_w_distances)
    print("-" * 38)
    print(f"Average 1D Wasserstein Distance: {avg_w_dist:.4f}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set plotting style
    sns.set_theme(style="whitegrid")

    # Prepare data for plotting
    features = [f'F_{i}' for i in range(1, 21)]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=features, y=feature_w_distances, palette="viridis")

    # Add labels and title
    plt.title('Univariate Wasserstein Distance per Feature (X_train vs X_test)', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Wasserstein Distance', fontsize=12)

    # Draw a line for the average Wasserstein distance
    plt.axhline(avg_w_dist, color='red', linestyle='--', label=f'Average Distance ({avg_w_dist:.4f})')
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set plotting style
    sns.set_theme(style="whitegrid")

    # Create a dedicated plot for target distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train.flatten(), kde=True, color='royalblue', alpha=0.6, label='y_train (Training Targets)', stat='density')
    sns.histplot(y_test.flatten(), kde=True, color='darkorange', alpha=0.4, label='y_test (Test Targets)', stat='density')

    # Add details
    plt.title('Comparison of Target Distributions (y_train vs y_test)', fontsize=16, fontweight='bold')
    plt.xlabel('Target Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.show()



    nonlin = "rf"
    error = "gaussian"
    hyper_path = "data/" + "Optim_hyperpar_" + nonlin + "_" + error + ".txt"
    hyperpars = open(hyper_path,'r').read()
    params_all = ast.literal_eval(hyperpars)
    params_elnet = params_all['Elastic Net']

    input_shape = (X_train.shape[1],)
    neurons_reg = 8
    learning_rate = 0.0005
    decay_factor = 300
    epochs = int(params_all['Artificial Neural Network']['epochs'])
    batch_size = int(params_all['Artificial Neural Network']['batch_size'])
    neurons_fe = int(params_all['Artificial Neural Network']['neurons_fe'])
    regularization = params_all['Artificial Neural Network']['regularization']
    dropout = params_all['Artificial Neural Network']['dropout']
    steps = int(X_train.shape[0]/batch_size)

    print(f"Number of features: {X_train.shape[1]}")
    print(f"Neurons in feature extractor: {neurons_fe}")

    # Train models: ML benchmarks
    elnet_mod = ElasticNet(**params_elnet)
    elnet_mod.fit(X_train, y_train)

    # Inspect scikit-learn ElasticNet model parameters
    print("=== Elastic Net Model Parameters ===")
    print(f"Intercept: {elnet_mod.intercept_[0]:.4f}")
    print("\nCoefficients:")
    for idx, coef in enumerate(elnet_mod.coef_):
        print(f"  Feature_{idx+1:<2}: {coef:.4f}")

    def feature_extraction(neurons_fe):
        model = Sequential()
        model.add(Dense(neurons_fe, input_shape = input_shape, kernel_regularizer = l2(regularization),
                            activation = "relu", kernel_constraint=MaxNorm(3)))
        model.add(Dropout(dropout))
        return model

    def regression(neurons_reg):
        model = Sequential()
        model.add(Dense(neurons_reg, activation = "relu", kernel_constraint=MaxNorm(3)))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation ="linear"))
        return model

    feature_extractor = feature_extraction(neurons_fe)
    regressor = regression(neurons_reg)
    inputs = Input(shape = input_shape)
    features = feature_extractor(inputs)
    outputs = regressor(features)
    n_source = X_train.shape[0] // batch_size

    baseline_model = Model(inputs=inputs, outputs=outputs)
    baseline_model.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor),
                            loss=["mse"], metrics = ["mse"])

    baseline_model.summary()
    print("Number of trainable parameters in baseline model:", baseline_model.count_params())

    for epoch in range(epochs):
        for i in range(steps):
            ind_source = np.random.randint(0, n_source)
            x_source_batch = X_train[ind_source * batch_size:(ind_source + 1) * batch_size]
            y_source_batch = y_train[ind_source * batch_size:(ind_source + 1) * batch_size]
            baseline_model.train_on_batch(x_source_batch, y_source_batch)

    # Predictions standard ML
    y_pred_elnet = np.expand_dims(elnet_mod.predict(X_test), axis=1)
    y_pred_baseline = baseline_model.predict(X_test)

    # Train models: WDGRL
    training_ratio = 1
    gen_rate = 0.0005
    gradient_penalty_weight = 1
    was_rate = 0.0005
    wd_param = 1

    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def make_wasserstein(neurons_fe):
        model = Sequential()
        model.add(Dense(int(neurons_reg),  input_shape = (int(neurons_fe),)))
        model.add(LeakyReLU())
        model.add(Dense(1))
        return model

    class RandomWeightedAverage(Layer):
        def __init__(self, batch_size):
            super().__init__()
            self.batch_size = batch_size

        def call(self, inputs):
            weights = K.random_uniform((self.batch_size, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    feature_extractor = feature_extraction(neurons_fe)
    regressor = regression(neurons_reg)
    inputs = Input(shape = input_shape)
    features = feature_extractor(inputs)
    outputs = regressor(features)

    reg_model = Model(inputs = inputs, outputs = outputs)
    reg_model.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor), loss = ["mse"], metrics = ["mse"])

    generator_input_for_wasserstein1 = Input(shape=input_shape)
    generator_input_for_wasserstein2 = Input(shape=input_shape)

    generated_samples_for_wasserstein1 = feature_extractor(generator_input_for_wasserstein1)
    generated_samples_for_wasserstein2 = feature_extractor(generator_input_for_wasserstein2)

    wasserstein = make_wasserstein(neurons_fe)

    wasserstein_output_from_generator1 = wasserstein(generated_samples_for_wasserstein1)
    wasserstein_output_from_generator2 = wasserstein(generated_samples_for_wasserstein2)

    regressor_output_from_generator1 = regressor(generated_samples_for_wasserstein1)

    averaged_samples = RandomWeightedAverage(batch_size)([generated_samples_for_wasserstein2, generated_samples_for_wasserstein1])
    averaged_samples_out = wasserstein(averaged_samples)
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=gradient_penalty_weight)
    partial_gp_loss.__name__ = "gradient_penalty"

    for layer in wasserstein.layers:
        layer.trainable = True
    wasserstein.trainable = True

    for layer in feature_extractor.layers:
        layer.trainable = False
    feature_extractor.trainable = False

    for layer in regressor.layers:
        layer.trainable = False
    regressor.trainable = False

    wasserstein_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1], outputs=[wasserstein_output_from_generator2, wasserstein_output_from_generator1, averaged_samples_out])
    wasserstein_model.compile(optimizer=Adam(was_rate, decay = was_rate/decay_factor), loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])

    for layer in wasserstein.layers:
        layer.trainable = False
    wasserstein.trainable = False

    for layer in feature_extractor.layers:
        layer.trainable = True
    feature_extractor.trainable = True

    for layer in regressor.layers:
        layer.trainable = True
    regressor.trainable = True

    generator_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1], outputs=[regressor_output_from_generator1, wasserstein_output_from_generator2, wasserstein_output_from_generator1])
    generator_model.compile(optimizer=Adam(gen_rate, decay = gen_rate/decay_factor), loss=["mse", wasserstein_loss, wasserstein_loss], loss_weights = [1, wd_param, wd_param])

    positive_y = np.ones((batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    n_source = X_train.shape[0] // batch_size
    n_target = X_test.shape[0] // batch_size

    for epoch in range(epochs):
        for i in range(steps):
            ind_source = np.random.randint(0, n_source)
            ind_target = np.random.randint(0, n_target)
            x_source_batch = X_train[ind_source * batch_size:(ind_source + 1) * batch_size]
            y_source_batch = y_train[ind_source * batch_size:(ind_source + 1) * batch_size]
            x_target_batch = X_test[ind_target * batch_size:(ind_target + 1) * batch_size]
            wasserstein_model.train_on_batch([x_target_batch, x_source_batch], [positive_y, negative_y, dummy_y])
            if i % training_ratio == 0:
                generator_model.train_on_batch([x_target_batch, x_source_batch], [y_source_batch, negative_y, positive_y])
    y_pred_wdgrl = reg_model.predict(X_test)

y_true = y_test
predictions = {
    "Elastic Net": y_pred_elnet,
    "Neural Network Baseline": y_pred_baseline,
    "WDGRL": y_pred_wdgrl
}

# 6.1 Actual versus predicted: closeness to the diagonal indicates better agreement.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
lims = [min(y_true.min(), *(p.min() for p in predictions.values())),
        max(y_true.max(), *(p.max() for p in predictions.values()))]
for ax, (name, pred) in zip(axes, predictions.items()):
    ax.scatter(y_true, pred, alpha=0.35, s=14)
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Ideal: prediction = actual")
    ax.set(title= "Actual vs Predicted (closeness to the diagonal indicates better agreement)", xlabel="Synthetic target (y_test)", ylabel="Predicted target")
    ax.legend()
fig.suptitle("Predicted versus synthetic target values", y=1.03, fontsize=15)
plt.tight_layout()
plt.show()


# 6.2 Residual distributions. Residual = prediction - observed target.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, pred) in zip(axes, predictions.items()):
    residual = pred - y_true
    sns.histplot(residual, kde=True, stat="density", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set(title=f"{name}: residual distribution", xlabel="Residual", ylabel="Density")
plt.tight_layout()
plt.show()


# 6.3 Residuals versus predictions: patterns/funnels can indicate bias or non-constant variance.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, pred) in zip(axes, predictions.items()):
    residual = pred - y_true
    ax.scatter(pred, residual, alpha=0.35, s=14)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set(title=name, xlabel="Predicted target", ylabel="Residual")
plt.tight_layout()
plt.show()



# 6.4 Compare scalar evaluation metrics. Lower MAE/RMSE is better; higher R² is better.
long_metrics = []
for name, pred in predictions.items():
    mae = np.mean(np.abs(pred - y_true))
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    r2 = 1 - np.sum((pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    long_metrics.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})
metrics_df = pd.DataFrame(long_metrics)

long_metrics = metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R²"],
                               var_name="Metric", value_name="Score")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
    subset = long_metrics[long_metrics["Metric"] == metric]
    sns.barplot(data=subset, x="Model", y="Score", ax=ax)
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=25)

plt.title("R² Model comparison ", y=1.04)
plt.suptitle("Model comparison on the synthetic target set", y=1.04)
plt.tight_layout()
plt.show()



# # 6.5 Training behavior: neural-network baseline and WDGRL total losses.
# baseline_epoch_mse = baseline_model.history.history["mse"]
# critic_losses = wasserstein_model.history.history["loss"]
# generator_losses = generator_model.history.history["loss"]

# fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
# axes[0].plot(np.arange(1, len(baseline_epoch_mse) + 1), baseline_epoch_mse, marker=".")
# axes[0].set(title="Neural-network baseline: training MSE", xlabel="Epoch", ylabel="Mean batch MSE")
# axes[1].plot(np.arange(1, len(critic_losses) + 1), critic_losses, label="Critic total loss")
# axes[1].plot(np.arange(1, len(generator_losses) + 1), generator_losses, label="Generator total loss")
# axes[1].set(title="WDGRL: total training losses", xlabel="Epoch", ylabel="Total loss")
# axes[1].legend()
# plt.tight_layout()
# plt.show()

# print("Interpretation note: adversarial critic/generator losses are not directly comparable to supervised MSE, "
#       "and their curves need not decrease monotonically. Check target-set metrics and residual plots as well.")




# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# x_test_by_theta = {theta: np.random.multivariate_normal(mean_train + theta * diff_vec, cov_test, size=test_size) for theta in thetas}
# axes[0].scatter(X_train[:, 0], X_train[:, 1], s=10, alpha=0.2, label="Source")
# for theta, samples in X_test_by_theta.items():
#     axes[0].scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.18, label=f"Target θ={theta}")
# axes[0].set(title="Feature-space shift (first two features)", xlabel="Feature 1", ylabel="Feature 2")
# axes[0].legend(markerscale=2, fontsize=8)

# theta_means = np.array([X_test_by_theta[t].mean(axis=0) for t in thetas])
# for feature_idx in range(min(5, X_train.shape[1])):
#     axes[1].plot(thetas, theta_means[:, feature_idx], marker="o", label=f"Feature {feature_idx+1}")
#     axes[1].axhline(X_train[:, feature_idx].mean(), linestyle=":", alpha=0.5)
# axes[1].set(title="Sample means as shift strength increases", xlabel="Shift strength θ", ylabel="Feature mean")
# axes[1].legend()
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")

# Create a dedicated plot for Comparison of Model Predictions

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
# Plotting the distributions of predictions from different models
sns.histplot(y_test.flatten(), kde=True, color='blue', alpha=0.5, label='True Predictions', stat='density')
sns.histplot(y_pred_elnet.flatten(), kde=True, color='green', alpha=0.5, label='Elastic Net Predictions', stat='density')
sns.histplot(y_pred_baseline.flatten(), kde=True, color='purple', alpha=0.5, label='ANN Baseline Predictions', stat='density')
sns.histplot(y_pred_wdgrl.flatten(), kde=True, color='red', alpha=0.5, label='WDGRL Predictions', stat='density')
# Add details
plt.title('Comparison of Model Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.show()