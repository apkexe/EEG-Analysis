import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import time
import os
from sklearn.utils import resample
import sklearn.utils
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Loading and splitting the dataset
def select_dataset():
    datasets = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    print("Available datasets: ")
    for idx, dataset in enumerate(datasets):
        print(f"{idx + 1}. {dataset}")
    
    choice = int(input("Select the dataset number you want to use: \nNOTE that if you choose a binary label dataset you need to modify the label inputs in the code accordingly!\n")) - 1
    if choice >= 0 and choice < len(datasets):
        return datasets[choice]
    else:
        print("Invalid choice. Please try again.")
        return select_dataset()
def plot_bootstrap_distribution(bootstrap_values, metric_name, model_name):
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_values, bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(bootstrap_values), color='red', linestyle='dashed', linewidth=1)
    plt.title(f'Bootstrap Distribution of {metric_name} for {model_name}')
    plt.xlabel(metric_name)
    plt.ylabel('Density')
    plt.show()

start = time.time()

# use the menu to select the dataset
selected_dataset = select_dataset()
print(f"Selected dataset: {selected_dataset}")
#dataset = pd.read_csv(selected_dataset)
dataset=pd.read_excel(selected_dataset)


y = dataset['label']  # 'label' = 0, 1, 2 corresponding to A, C, F
X = dataset.drop('label', axis=1) # keeps the data without the label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # to vlepoume to 0.3

# clf1 = RandomForestClassifier(random_state=42)
# clf2 = KNeighborsClassifier()
# clf3 = SVC(random_state=42)
# clf4 = LGBMClassifier(random_state=42)
# clf5 = XGBClassifier(random_state=42)

# # Building the pipelines
# pipe1 = Pipeline([('std', StandardScaler()),
#                   ('clf1', clf1)])

# pipe2 = Pipeline([('std', StandardScaler()),
#                   ('clf2', clf2)])
# pipe3 = Pipeline([('std', StandardScaler()),
#                   ('clf2', clf2)])

# pipe4 = Pipeline([('std', StandardScaler()),
#                   ('clf4', clf4)])

# pipe4 = Pipeline([('std', StandardScaler()),
#                   ('clf4', clf4)])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', MinMaxScaler()) # scales our data within (0,1)
        ]), X.columns)
    ])

# roaming through desired classifiers and stating their hyperparameter grids
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }),
    'KNN': (KNeighborsClassifier(), {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    }),
    'SVM': (SVC(random_state=42), {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }),
    # # 'LightGBM': (LGBMClassifier(random_state=42), {
    # # 'classifier__n_estimators': [100, 150],
    # # 'classifier__max_depth': [ 6, 9, -1],
    # # 'classifier__learning_rate': [0.01, 0.1],
    # # 'classifier__min_data_in_leaf': [20]
    # # }),
    'XGBoost': (XGBClassifier(random_state=42), {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6, 9],
        'classifier__learning_rate': [0.01, 0.1, 0.3]
    })
}
#cod form github

gridcvs = {}
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

for name, (model, param_grid) in classifiers.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    gcv = GridSearchCV(estimator=pipeline,
                       param_grid=param_grid,
                       scoring='accuracy',
                       n_jobs=-1,
                       cv=inner_cv,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, 
                                   X=X_train, 
                                   y=y_train, 
                                   cv=outer_cv,
                                   n_jobs=-1)
    print('%s | outer ACC %.2f%% +/- %.2f' % 
          (name, nested_score.mean() * 100, nested_score.std() * 100))
print("\n")
print("-------------------------------------------")
print("Final choices of models by trial on test set: \n")
results = []
bootstrap_iterations = 1000
for name, (model, param_grid) in classifiers.items():
    # Create the pipeline with preprocessing + classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    gcv_model_select = GridSearchCV(estimator=pipeline,
                                param_grid=param_grid,
                                scoring='accuracy',
                                n_jobs=-1,
                                cv=inner_cv,
                                verbose=1,
                                refit=True)
    gcv_model_select.fit(X_train, y_train) # fitting the model
    print(f'{name} best params: {gcv_model_select.best_params_}')
    best_model = gcv_model_select.best_estimator_
    ## We can skip the next step because we set refit=True
    ## so scikit-learn has already fit the model to the
    ## whole training set

    # best_model.fit(X_train, y_train)


    train_acc = accuracy_score(y_true=y_train, y_pred=best_model.predict(X_train))
    test_acc = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))

    #print('Accuracy %.2f%% (average over k-fold CV test folds)' %
          #(100 * gcv_model_select.best_score_))
    #print('Best Parameters: %s' % gcv_model_select.best_params_)

    print('Training Accuracy: %.2f%%' % (100 * train_acc))
    print('Test Accuracy: %.2f%%' % (100 * test_acc))
    print("\n")
    #bootstrap_part
    
    bootstrap_accuracies = []
    bootstrap_f1s = []
    bootstrap_sensitivities = []
    bootstrap_specificities = []
    bootstrap_x_samples_mean=[]
    bootstrap_y_samples_mean=[]
    

    print(f"Performing bootstrap evaluation for {name}...")
    for i in range(bootstrap_iterations):
        X_boot, y_boot = sklearn.utils.resample(X_train,y_train, replace=True, n_samples=None, random_state=i, stratify=y_train)
        bootstrap_x_samples_mean.append(np.mean(X_boot.iloc[:,3]))
        bootstrap_y_samples_mean.append(np.mean(y_boot))
        

        boot_model = clone(best_model)
        boot_model.fit(X_boot, y_boot)
        y_boot_pred = boot_model.predict(X_test)

        boot_cm = confusion_matrix(y_test, y_boot_pred, labels=[0, 1])
        tp, fn, fp, tn = boot_cm.ravel()

        bootstrap_accuracies.append(accuracy_score(y_test, y_boot_pred))
        bootstrap_f1s.append(f1_score(y_test, y_boot_pred, average='weighted'))
        bootstrap_sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        bootstrap_specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    plot_bootstrap_distribution(bootstrap_accuracies, 'Accuracy', name)
    plot_bootstrap_distribution(bootstrap_x_samples_mean, 'X_samples_mean', name)
    plot_bootstrap_distribution(bootstrap_y_samples_mean, 'Y_samples_mean', name)
    #plot_bootstrap_distribution(bootstrap_f1s, 'F1 Score', name)

    results.append({
         'Dataset': selected_dataset,
         'Model': name,
         'Training Accuracy': f'{100 * train_acc:.2f}%',  # Using f-string for formatting
         'Test Accuracy': f'{100 * test_acc:.2f}%',
         'Bootsrap Test Accuracy': f"{np.mean(bootstrap_accuracies):.4f} ± {np.std(bootstrap_accuracies):.4f}",
         'F1 Score': f"{np.mean(bootstrap_f1s):.4f} ± {np.std(bootstrap_f1s):.4f}",
         'Sensitivity': f"{np.mean(bootstrap_sensitivities):.4f} ± {np.std(bootstrap_sensitivities):.4f}",
         'Specificity': f"{np.mean(bootstrap_specificities):.4f} ± {np.std(bootstrap_specificities):.4f}"
     })
#save results
results_df = pd.DataFrame(results)
excel_file = 'new_models_results.xlsx'
if os.path.exists(excel_file):
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Results', startrow=writer.sheets['Results'].max_row, header=False)
else:
    results_df.to_excel(excel_file, index=False, sheet_name='Results')


end = time.time()
total = (end - start)/60
print(f"Total elapsed time: {total} minutes") 