import json
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Create the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Titanic Survival Prediction\n",
                "\n",
                "This notebook contains the model development for predicting survival on the Titanic dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.metrics import accuracy_score, classification_report\n",
                "import joblib\n",
                "import warnings\n",
                "import os\n",
                "warnings.filterwarnings('ignore')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Loading and Exploration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the dataset from local file\n",
                "df = pd.read_csv('../data/titanic.csv')\n",
                "\n",
                "# Display basic information\n",
                "print(\"Dataset Shape:\", df.shape)\n",
                "print(\"\\nMissing Values:\")\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# Display first few rows\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Feature Engineering"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def engineer_features(df):\n",
                "    # Create a copy to avoid modifying the original dataframe\n",
                "    df = df.copy()\n",
                "    \n",
                "    # Extract titles from names\n",
                "    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)\n",
                "    \n",
                "    # Create family size feature\n",
                "    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
                "    \n",
                "    # Create fare bins\n",
                "    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Mid', 'Mid-High', 'High'])\n",
                "    \n",
                "    # Create age bins\n",
                "    df['AgeBin'] = pd.qcut(df['Age'], 4, labels=['Young', 'Adult', 'Middle', 'Senior'])\n",
                "    \n",
                "    # Fill missing values\n",
                "    df['Age'].fillna(df['Age'].median(), inplace=True)\n",
                "    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n",
                "    \n",
                "    # Convert categorical variables\n",
                "    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin'])\n",
                "    \n",
                "    return df\n",
                "\n",
                "# Apply feature engineering\n",
                "df_engineered = engineer_features(df)\n",
                "df_engineered.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Training and Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare features and target\n",
                "features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize'] + \\\n",
                "           [col for col in df_engineered.columns if col.startswith(('Sex_', 'Embarked_', 'Title_', 'FareBin_', 'AgeBin_'))]\n",
                "X = df_engineered[features]\n",
                "y = df_engineered['Survived']\n",
                "\n",
                "# Split the data\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "\n",
                "# Scale the features\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "# Train Random Forest model\n",
                "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                "rf_model.fit(X_train_scaled, y_train)\n",
                "\n",
                "# Make predictions\n",
                "y_pred = rf_model.predict(X_test_scaled)\n",
                "\n",
                "# Print results\n",
                "print(\"Random Forest Model Performance:\")\n",
                "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
                "print(\"\\nClassification Report:\")\n",
                "print(classification_report(y_test, y_pred))\n",
                "\n",
                "# Create model directory if it doesn't exist\n",
                "os.makedirs('../model', exist_ok=True)\n",
                "\n",
                "# Save the model and scaler\n",
                "joblib.dump(rf_model, '../model/rf_model.pkl')\n",
                "joblib.dump(scaler, '../model/scaler.pkl')\n",
                "\n",
                "# Plot feature importance\n",
                "plt.figure(figsize=(10, 6))\n",
                "feature_importance = pd.DataFrame({\n",
                "    'feature': features,\n",
                "    'importance': rf_model.feature_importances_\n",
                "})\n",
                "feature_importance = feature_importance.sort_values('importance', ascending=False)\n",
                "sns.barplot(x='importance', y='feature', data=feature_importance.head(10))\n",
                "plt.title('Top 10 Most Important Features')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('notebooks/titanic_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook created successfully at notebooks/titanic_model.ipynb") 