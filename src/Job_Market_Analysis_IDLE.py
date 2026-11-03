# ============================================================
# GLOBAL JOB MARKET ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# 1. LOAD THE FOUR DATASETS
# ============================================================

jobs = pd.read_csv('/Users/sachinraj/Desktop/Python II/Project/archive/jobs.csv')
company = pd.read_csv('/Users/sachinraj/Desktop/Python II/Project/archive/jobs_by_company.csv')
daily = pd.read_csv('/Users/sachinraj/Desktop/Python II/Project/archive/open_jobs_daily.csv')
skills = pd.read_csv('/Users/sachinraj/Desktop/Python II/Project/archive/skill_demand_daily.csv')


print("\n--- JOBS DATASET ---")
print(jobs.head())
print(jobs.info())

print("\n--- COMPANY DATASET ---")
print(company.head())

print("\n--- DAILY JOBS DATASET ---")
print(daily.head())

print("\n--- SKILL DEMAND DATASET ---")
print(skills.head())


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Remove duplicate rows from each dataset
jobs = jobs.drop_duplicates()
company = company.drop_duplicates()
daily = daily.drop_duplicates()
skills = skills.drop_duplicates()

# Remove rows without job title or category
jobs = jobs.dropna(subset=['title', 'category'])

# Clean text by converting it to lowercase and removing spaces
jobs['title'] = jobs['title'].astype(str).str.lower().str.strip()
jobs['category'] = jobs['category'].astype(str).str.lower().str.strip()

# Remove blank titles/categories
jobs = jobs[(jobs['title'] != '') & (jobs['category'] != '')]

# Remove the general 'other' category
jobs = jobs[jobs['category'] != 'other']

print("\nJobs dataset shape after cleaning:", jobs.shape)
print("\nJob categories:")
print(jobs['category'].value_counts())


# ============================================================
# OBJECTIVE 1: FIND THE MOST COMMON JOB CATEGORIES
# Dataset: jobs.csv
# ============================================================

print("\nOBJECTIVE 1: Most common job categories")

category_count = jobs['category'].value_counts().head(10)
print(category_count)

category_count.plot(kind='bar', figsize=(9, 5))
plt.title('Top 10 Job Categories')
plt.xlabel('Job Category')
plt.ylabel('Number of Job Listings')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 2: FIND CATEGORIES WITH FRESHER-FRIENDLY JOBS
# Dataset: jobs.csv
# ============================================================

print("\nOBJECTIVE 2: Fresher-friendly jobs by category")

if 'fresher_friendly' in jobs.columns:
    # Keep records marked as fresher-friendly
    fresher_jobs = jobs[
        jobs['fresher_friendly'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )
    ]

    fresher_count = fresher_jobs['category'].value_counts().head(10)
    print(fresher_count)

    fresher_count.plot(kind='bar', figsize=(9, 5))
    plt.title('Top Categories for Fresher-Friendly Jobs')
    plt.xlabel('Job Category')
    plt.ylabel('Number of Jobs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("The fresher_friendly column was not found in jobs.csv.")


# ============================================================
# OBJECTIVE 3: FIND COMPANIES WITH THE MOST FRESHER ROLES
# Dataset: jobs_by_company.csv
# ============================================================

print("\nOBJECTIVE 3: Companies offering fresher roles")

if 'company' in company.columns and 'fresher_roles' in company.columns:
    company['fresher_roles'] = pd.to_numeric(
        company['fresher_roles'], errors='coerce'
    )
    company = company.dropna(subset=['fresher_roles'])
    top_companies = company.sort_values(
        by='fresher_roles', ascending=False
    ).head(10)

    print(top_companies[['company', 'fresher_roles']])

    plt.figure(figsize=(10, 5))
    plt.bar(top_companies['company'].astype(str),
            top_companies['fresher_roles'])
    plt.title('Top 10 Companies Offering Fresher Roles')
    plt.xlabel('Company')
    plt.ylabel('Number of Fresher Roles')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()
else:
    print("Expected columns 'company' and 'fresher_roles' were not found.")
    print("Available columns:", list(company.columns))


# ============================================================
# OBJECTIVE 4: VISUALISE DAILY OPEN-JOB TRENDS
# Dataset: open_jobs_daily.csv
# ============================================================

print("\nOBJECTIVE 4: Daily open-job trends")

if 'date' in daily.columns and 'open_jobs' in daily.columns:
    daily['date'] = pd.to_datetime(daily['date'], errors='coerce')
    daily['open_jobs'] = pd.to_numeric(daily['open_jobs'], errors='coerce')
    daily = daily.dropna(subset=['date', 'open_jobs'])
    daily = daily.sort_values('date')

    plt.figure(figsize=(10, 5))
    plt.plot(daily['date'], daily['open_jobs'], marker='o')
    plt.title('Daily Open Job Trend')
    plt.xlabel('Date')
    plt.ylabel('Number of Open Jobs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if 'fresher_friendly_jobs' in daily.columns:
        daily['fresher_friendly_jobs'] = pd.to_numeric(
            daily['fresher_friendly_jobs'], errors='coerce'
        )
        plt.figure(figsize=(10, 5))
        plt.plot(daily['date'], daily['open_jobs'], label='All Open Jobs')
        plt.plot(daily['date'], daily['fresher_friendly_jobs'],
                 label='Fresher-Friendly Jobs')
        plt.title('Daily Open Jobs: All vs Fresher-Friendly')
        plt.xlabel('Date')
        plt.ylabel('Number of Jobs')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
else:
    print("Expected columns 'date' and 'open_jobs' were not found.")
    print("Available columns:", list(daily.columns))


# ============================================================
# OBJECTIVE 5: FIND THE MOST DEMANDED SKILLS
# Dataset: skill_demand_daily.csv
# ============================================================

print("\nOBJECTIVE 5: Most demanded skills")

if 'skill' in skills.columns and 'jobs' in skills.columns:
    skills['jobs'] = pd.to_numeric(skills['jobs'], errors='coerce')
    skills = skills.dropna(subset=['skill', 'jobs'])
    top_skills = skills.groupby('skill')['jobs'].sum().sort_values(
        ascending=False
    ).head(10)

    print(top_skills)

    top_skills.sort_values().plot(kind='barh', figsize=(9, 5))
    plt.title('Top 10 Most Demanded Skills')
    plt.xlabel('Number of Jobs')
    plt.ylabel('Skill')
    plt.tight_layout()
    plt.show()
else:
    print("Expected columns 'skill' and 'jobs' were not found.")
    print("Available columns:", list(skills.columns))


# ============================================================
# OBJECTIVE 6: COMPARE REMOTE AND NON-REMOTE JOBS
# Dataset: jobs.csv
# ============================================================

print("\nOBJECTIVE 6: Remote and non-remote jobs")

if 'remote' in jobs.columns:
    remote_count = jobs['remote'].astype(str).value_counts()
    print(remote_count)

    remote_count.plot(kind='bar', figsize=(7, 4))
    plt.title('Remote vs Other Job Listings')
    plt.xlabel('Remote Value')
    plt.ylabel('Number of Jobs')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("The remote column was not found in jobs.csv.")


# ============================================================
# OBJECTIVE 7: PREDICT JOB CATEGORY USING MACHINE LEARNING
# Input (X): Job title
# Output (y): Job category
# ============================================================

print("\nOBJECTIVE 7: Machine Learning - Job Category Prediction")

# Use the eight most common categories to keep training manageable
top_categories = jobs['category'].value_counts().head(8).index
ml_data = jobs[jobs['category'].isin(top_categories)][
    ['title', 'category']
].copy()

# Use up to 2,500 rows to keep the program manageable in IDLE
ml_data = ml_data.sample(
    n=min(2500, len(ml_data)),
    random_state=42
)

print("\nMachine Learning data shape:", ml_data.shape)
print("\nCategories used:")
print(ml_data['category'].value_counts())

# Independent variable: job title
X = ml_data['title']

# Dependent variable: job category
y = ml_data['category']

# Split the dataset into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Convert job-title text into numeric word-count features
vectorizer = CountVectorizer(max_features=300, stop_words='english')
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

print("\nMODEL 1: LOGISTIC REGRESSION")

lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train, y_train)

# Predict job categories for test data
lr_pred = lr_model.predict(X_test)

# Calculate and display accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Accuracy:", round(lr_accuracy, 4))
print("Accuracy (%):", round(lr_accuracy * 100, 2))

# Display confusion matrix as numbers (not a graph)
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred, labels=lr_model.classes_))


# ============================================================
# MODEL 2: NAIVE BAYES
# ============================================================

print("\nMODEL 2: NAIVE BAYES")

nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict job categories for test data
nb_pred = nb_model.predict(X_test)

# Calculate and display accuracy
nb_accuracy = accuracy_score(y_test, nb_pred)
print("Accuracy:", round(nb_accuracy, 4))
print("Accuracy (%):", round(nb_accuracy * 100, 2))

# Display confusion matrix as numbers (not a graph)
print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred, labels=nb_model.classes_))


# ============================================================
# MODEL 3: DECISION TREE CLASSIFIER
# Follows the structure used in the classroom example.
# ============================================================

print("\nMODEL 3: DECISION TREE CLASSIFIER")

# Create Decision Tree Classifier
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42,
    max_depth=6
)

# Train the model
dt_model.fit(X_train, y_train)

# Predict test data
dt_pred = dt_model.predict(X_test)

# Confusion Matrix
dt_cm = confusion_matrix(y_test, dt_pred, labels=dt_model.classes_)

print("Confusion Matrix:")
print(dt_cm)

# Accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Accuracy:", dt_accuracy)
print("Accuracy (%):", dt_accuracy * 100)

# Visualise the Decision Tree
# max_depth=3 limits the displayed tree to make it easier to read.
# It does not change the trained model.
plt.figure(figsize=(22, 10))

plot_tree(
    dt_model,
    max_depth=3,
    feature_names=vectorizer.get_feature_names_out(),
    class_names=[str(name) for name in dt_model.classes_],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.show()


# ============================================================
# MODEL 4: RANDOM FOREST CLASSIFIER
# ============================================================

print("\nMODEL 4: RANDOM FOREST CLASSIFIER")

rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict test data
rf_pred = rf_model.predict(X_test)

# Calculate and display accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy:", round(rf_accuracy, 4))
print("Accuracy (%):", round(rf_accuracy * 100, 2))

# Display confusion matrix as numbers (not a graph)
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred, labels=rf_model.classes_))


# ============================================================
# MODEL 5: K-NEAREST NEIGHBOURS (KNN)
# ============================================================

print("\nMODEL 5: K-NEAREST NEIGHBOURS")

knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict test data
knn_pred = knn_model.predict(X_test)

# Calculate and display accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)
print("Accuracy:", round(knn_accuracy, 4))
print("Accuracy (%):", round(knn_accuracy * 100, 2))

# Display confusion matrix as numbers (not a graph)
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred, labels=knn_model.classes_))


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Naive Bayes',
        'Decision Tree',
        'Random Forest',
        'KNN'
    ],
    'Accuracy': [
        lr_accuracy,
        nb_accuracy,
        dt_accuracy,
        rf_accuracy,
        knn_accuracy
    ]
})

comparison = comparison.sort_values(
    by='Accuracy',
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# Save results to a CSV file
comparison.to_csv('model_comparison.csv', index=False)

# Compare accuracy of all five models in a bar chart
plt.figure(figsize=(9, 5))
plt.bar(comparison['Model'], comparison['Accuracy'] * 100)
plt.title('Accuracy Comparison of Five ML Models')
plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Display the best model according to accuracy
best_model = comparison.iloc[0]['Model']
best_accuracy = comparison.iloc[0]['Accuracy']

print("\nBest model based on accuracy:", best_model)
print("Best accuracy (%):", round(best_accuracy * 100, 2))

print("\nPROJECT COMPLETED")
