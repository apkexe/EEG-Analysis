import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

def read_all_csv_in_directory(directory):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    return csv_files

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='train_accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def train_model(X_train, y_train, X_val, y_val, num_nodes, dropout_prob, lr, batch_size, epochs):
    input_shape = X_train.shape[1]
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = nn_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    return nn_model, history

def save_results_to_excel(dataset_name, results, hyperparams, file_path="MLP_results.xlsx"):
    try:
        new_row = pd.DataFrame({
            'Dataset': [dataset_name],
            'Nodes': [hyperparams['nodes']],
            'Dropout': [hyperparams['dropout']],
            'Learning Rate': [hyperparams['lr']],
            'Batch Size': [hyperparams['batch_size']],
            'Accuracy': [results['accuracy']],
            'Precision (Class 0)': [results['0']['precision']],
            'Precision (Class 1)': [results['1']['precision']],
            'Recall (Class 0)': [results['0']['recall']],
            'Recall (Class 1)': [results['1']['recall']],
            'F1-Score (Class 0)': [results['0']['f1-score']],
            'F1-Score (Class 1)': [results['1']['f1-score']]
        })

        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path, sheet_name='Results')
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)

            with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
                updated_df.to_excel(writer, index=False, sheet_name='Results')
        else:
            new_row.to_excel(file_path, index=False, sheet_name='Results')

        print(f"Results successfully saved to {file_path}")

    except Exception as e:
        print(f"An error occurred while saving to Excel: {e}")

# Main script
np.random.seed(42)  # Set a fixed random seed for reproducibility
tf.random.set_seed(42)

base_path = r'C:\Users\apkexe\Desktop\multiclass\AFC_unfiltered'
csv_files = read_all_csv_in_directory(base_path)

for csv_file in csv_files:
    print(f"Processing {csv_file}")

    df = pd.read_csv(csv_file, on_bad_lines='skip')

    train, valid, test = np.split(df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))])

    train, X_train, y_train = scale_dataset(train, oversample=True)
    valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
    test, X_test, y_test = scale_dataset(test, oversample=False)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"X_test shape: {X_test.shape}")

    epochs = 50
    dropout_prob = 0.4
    lr = 0.01

    for num_nodes in [16, 32, 64]:
        for batch_size in [32, 64, 128]:
            print(f"Training with {num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
            model, history = train_model(X_train, y_train, X_valid, y_valid, num_nodes, dropout_prob, lr, batch_size, epochs)
            # plot_history(history)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int).reshape(-1,)

            # Calculate individual metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, average=None)
            recall = recall_score(y_test, y_pred_binary, average=None)
            f1 = f1_score(y_test, y_pred_binary, average=None)

            report = classification_report(y_test, y_pred_binary, output_dict=True)

            # Add the accuracy to the report
            report['accuracy'] = accuracy

            # Save the results for each hyperparameter combination
            hyperparams = {'nodes': num_nodes, 'dropout': dropout_prob, 'lr': lr, 'batch_size': batch_size}
            save_results_to_excel(os.path.basename(csv_file), report, hyperparams, file_path="MLP_results_AFC_unfiltered.xlsx")

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1-score: {f1}")
            print(classification_report(y_test, y_pred_binary))