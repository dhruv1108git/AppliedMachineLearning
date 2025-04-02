#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# ### Data Preparation
# 
# - Loads and preprocesses SMS data.  
# - Converts labels to binary (spam = 1, ham = 0).  
# - Cleans text by removing special characters, stopwords, and extra spaces.  
# - Removes duplicate messages.  
# - Handles outliers based on message length.  
# - Performs exploratory data analysis (EDA) with visualizations.  
# - Prepares features like text length and word count.  
# - Splits data into train, validation, and test sets.  
# - Analyzes correlations between features and labels.  
# - Saves processed data for further use.

# In[2]:


class DataPreparation:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def load_data(self, filepath):
        """Load SMS data from given filepath"""
        try:
            # Read data with tab separator
            df = pd.read_csv(filepath, sep='\t', names=['label', 'message'])

            # Verify data format
            print("\nInitial data shape:", df.shape)
            print("\nSample of loaded data:")
            print(df.head())

            # Convert labels to standard format
            df['label'] = df['label'].str.lower()  # Convert to lowercase
            if not all(df['label'].isin(['ham', 'spam'])):
                raise ValueError("Labels must be either 'ham' or 'spam'")

            # Convert labels to numeric (0 for ham, 1 for spam)
            df['label'] = (df['label'] == 'spam').astype(int)

            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print("\nWarning: Found missing values:")
                print(missing[missing > 0])

                # Fill missing messages with empty string
                df['message'] = df['message'].fillna('')

            print(f"\nLoaded {len(df)} messages")
            print("\nLabel distribution:")
            print(df['label'].value_counts(normalize=True))
            print("\nLabel mapping: 0=ham, 1=spam")

            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def clean_text(self, text):
        """Clean text data"""
        try:
            # Handle empty or NaN values
            if pd.isna(text) or str(text).strip() == '':
                print(f"Warning: Empty text found")
                return ''

            # Convert to string and lowercase
            text = str(text).lower()

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)

            # Remove extra whitespace
            text = ' '.join(text.split())

            # Remove stopwords
            if text:  # Only if text is not empty
                words = text.split()
                words = [w for w in words if w not in self.stop_words]
                text = ' '.join(words)

            # If after cleaning text is empty, return placeholder
            if not text.strip():
                return 'empty_message'

            return text
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return 'error_in_cleaning'

    def remove_duplicates(self, df):
        """Remove duplicate messages"""
        initial_size = len(df)
        df = df.drop_duplicates(subset=['message'])
        final_size = len(df)
        print(f"Removed {initial_size - final_size} duplicate messages")
        return df

    def handle_outliers(self, df):
        """Handle outlier messages based on length"""
        df['message_length'] = df['message'].str.len()
        Q1 = df['message_length'].quantile(0.25)
        Q3 = df['message_length'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df['message_length'] >= Q1 - 1.5 * IQR) &
            (df['message_length'] <= Q3 + 1.5 * IQR)
        ]
        df = df.drop('message_length', axis=1)
        return df

    def analyze_data(self, df):
        """Perform exploratory data analysis"""
        print("\nPerforming EDA...")

        # Message length distribution
        df['message_length'] = df['message'].str.len()

        # Plot distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Label distribution
        sns.countplot(x='label', data=df, ax=ax1)
        ax1.set_title('Distribution of Labels')

        # Message length by label
        sns.boxplot(x='label', y='message_length', data=df, ax=ax2)
        ax2.set_title('Message Length by Label')

        plt.tight_layout()
        plt.savefig('eda_plots.png')
        plt.close()

        # Print statistics
        print("\nLabel Distribution:")
        print(df['label'].value_counts(normalize=True))

        print("\nMessage Length Statistics:")
        print(df.groupby('label')['message_length'].describe())

        df = df.drop('message_length', axis=1)
        return df

    def prepare_features(self, df):
        """Prepare features"""
        print("\nPreparing features...")

        # Ensure message column is string type
        df['message'] = df['message'].fillna('').astype(str)

        # Clean messages
        print("Cleaning text data...")
        df['processed_text'] = df['message'].apply(self.clean_text)

        # Check for empty processed texts
        empty_texts = df['processed_text'].str.strip() == ''
        if empty_texts.any():
            print(f"Warning: Found {empty_texts.sum()} empty texts after processing")
            # Replace empty texts with placeholder
            df.loc[empty_texts, 'processed_text'] = 'empty_message'

        # Verify processed data
        print("\nSample of processed texts:")
        print(df[['message', 'processed_text']].head())

        # Count unique processed texts
        n_unique = df['processed_text'].nunique()
        print(f"\nNumber of unique processed texts: {n_unique}")

        return df

    def split_data(self, df, train_size=0.7, val_size=0.15, random_state=42):
        """Split data into train/validation/test sets"""
        print("\nSplitting data...")

        # Verify data before splitting
        if df['processed_text'].isna().any():
            raise ValueError("Found NaN values in processed_text")

        if df['label'].isna().any():
            raise ValueError("Found NaN values in labels")

        # Verify label distribution
        label_dist = df['label'].value_counts()
        print("\nLabel distribution before splitting:")
        print(label_dist)

        if len(label_dist) != 2:
            raise ValueError(f"Expected 2 classes, found {len(label_dist)}")

        # Perform stratified split
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=df['label'],
            random_state=random_state
        )

        # Split temp into validation and test
        val_size_adjusted = val_size / (1 - train_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_adjusted,
            stratify=temp_df['label'],
            random_state=random_state
        )

        # Verify splits
        print(f"\nTrain set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")

        # Verify label distribution in splits
        print("\nLabel distribution in splits:")
        print("Train:", train_df['label'].value_counts(normalize=True))
        print("Validation:", val_df['label'].value_counts(normalize=True))
        print("Test:", test_df['label'].value_counts(normalize=True))

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df):
        """Save the data splits"""
        # Final verification
        for name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            # Check for required columns
            if not {'label', 'processed_text'}.issubset(df.columns):
                raise ValueError(f"Missing required columns in {name} set")

            # Check for empty texts
            empty = df['processed_text'].str.strip() == ''
            if empty.any():
                print(f"Warning: Found {empty.sum()} empty texts in {name} set")

            # Check label distribution
            print(f"\n{name.capitalize()} set label distribution:")
            print(df['label'].value_counts(normalize=True))

        # Save to CSV
        train_df.to_csv('train.csv', index=False)
        val_df.to_csv('validation.csv', index=False)
        test_df.to_csv('test.csv', index=False)

        print("\nData splits saved successfully")

    def analyze_correlations(self, df):
        """Analyze feature correlations and interactions"""
        print("\nAnalyzing feature correlations...")

        # Create basic text features
        df['text_length'] = df['processed_text'].str.len()
        df['word_count'] = df['processed_text'].str.split().str.len()
        df['avg_word_length'] = df['text_length'] / df['word_count']

        # Analyze correlations with target
        numeric_features = ['text_length', 'word_count', 'avg_word_length']
        correlations = df[numeric_features + ['label']].corr()['label'].sort_values()

        plt.figure(figsize=(10, 6))
        correlations.plot(kind='bar')
        plt.title('Feature Correlations with Target')
        plt.tight_layout()
        plt.show()

        return df

    def run_preparation(self, filepath):
        """Run the full data preparation pipeline"""
        print("Starting data preparation...")

        # Load data
        df = self.load_data(filepath)

        # Remove duplicates
        df = self.remove_duplicates(df)

        # Handle outliers
        df = self.handle_outliers(df)

        # Analyze data
        df = self.analyze_data(df)

        # Prepare features
        df = self.prepare_features(df)

        # Add correlation analysis
        df = self.analyze_correlations(df)

        # Split data
        train_df, val_df, test_df = self.split_data(df)

        # Save splits
        self.save_splits(train_df, val_df, test_df)

        return train_df, val_df, test_df

# In[6]:


data_prep = DataPreparation()
train_df, val_df, test_df = data_prep.run_preparation("SMSSpamCollection")

# ========== DVC Implementation ==========
import os
import subprocess

# Function to run shell commands and print output
def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result

# Save the raw data as raw_data.csv
def save_raw_data():
    # Read the original data and save as raw_data.csv
    raw_data = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'message'])
    raw_data.to_csv("raw_data.csv", index=False)
    print("Raw data saved to raw_data.csv")

# Initialize DVC
def init_dvc():
    # Initialize DVC if not already initialized
    if not os.path.exists(".dvc"):
        run_command("dvc init")
        print("DVC initialized")
    else:
        print("DVC already initialized")
    
    # Add raw data to DVC
    run_command("dvc add raw_data.csv")

# Process data and create a split with the first random seed (42)
def create_first_version():
    print("\n\n==== Creating first version (random_state=42) ====")
    data_prep = DataPreparation()
    # Load raw data
    df = data_prep.load_data("SMSSpamCollection")
    
    # Save raw data
    df.to_csv("raw_data.csv", index=False)
    
    # Process data
    df = data_prep.remove_duplicates(df)
    df = data_prep.handle_outliers(df)
    df = data_prep.analyze_data(df)
    df = data_prep.prepare_features(df)
    df = data_prep.analyze_correlations(df)
    
    # Split data with random_state=42
    train_df, val_df, test_df = data_prep.split_data(df, random_state=42)
    
    # Save splits
    data_prep.save_splits(train_df, val_df, test_df)
    
    # Add to DVC
    run_command("dvc add train.csv validation.csv test.csv")
    
    # Commit the first version
    run_command("git add .gitignore raw_data.csv.dvc train.csv.dvc validation.csv.dvc test.csv.dvc")
    run_command('git commit -m "First data version with random_state=42"')
    
    print("\nFirst version created and committed")

# Process data and create a split with a different random seed (123)
def create_second_version():
    print("\n\n==== Creating second version (random_state=123) ====")
    data_prep = DataPreparation()
    
    # Load and process data (reuse raw data)
    df = pd.read_csv("raw_data.csv")
    
    # Process data
    df = data_prep.remove_duplicates(df)
    df = data_prep.handle_outliers(df)
    df = data_prep.analyze_data(df)
    df = data_prep.prepare_features(df)
    df = data_prep.analyze_correlations(df)
    
    # Split data with random_state=123
    train_df, val_df, test_df = data_prep.split_data(df, random_state=123)
    
    # Save splits
    data_prep.save_splits(train_df, val_df, test_df)
    
    # Update DVC
    run_command("dvc add train.csv validation.csv test.csv")
    
    # Commit the second version
    run_command("git add train.csv.dvc validation.csv.dvc test.csv.dvc")
    run_command('git commit -m "Second data version with random_state=123"')
    
    print("\nSecond version created and committed")

# Checkout the first version and print distributions
def checkout_first_version():
    print("\n\n==== Checking out first version (random_state=42) ====")
    # Get the first commit hash
    result = subprocess.run('git log --oneline | grep "First data version" | cut -d " " -f 1', 
                           shell=True, capture_output=True, text=True)
    commit_hash = result.stdout.strip()
    
    # Checkout the data from the first version
    run_command(f"git checkout {commit_hash} -- train.csv.dvc validation.csv.dvc test.csv.dvc")
    run_command("dvc checkout train.csv validation.csv test.csv")
    
    # Load and print distributions
    print("\nDistribution of target variable in first version:")
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")
    
    print("\nTrain.csv distribution:")
    print(train_df['label'].value_counts())
    
    print("\nValidation.csv distribution:")
    print(val_df['label'].value_counts())
    
    print("\nTest.csv distribution:")
    print(test_df['label'].value_counts())

# Checkout the second version and print distributions
def checkout_second_version():
    print("\n\n==== Checking out second version (random_state=123) ====")
    # Get the second commit hash
    result = subprocess.run('git log --oneline | grep "Second data version" | cut -d " " -f 1', 
                           shell=True, capture_output=True, text=True)
    commit_hash = result.stdout.strip()
    
    # Checkout the data from the second version
    run_command(f"git checkout {commit_hash} -- train.csv.dvc validation.csv.dvc test.csv.dvc")
    run_command("dvc checkout train.csv validation.csv test.csv")
    
    # Load and print distributions
    print("\nDistribution of target variable in second version:")
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")
    
    print("\nTrain.csv distribution:")
    print(train_df['label'].value_counts())
    
    print("\nValidation.csv distribution:")
    print(val_df['label'].value_counts())
    
    print("\nTest.csv distribution:")
    print(test_df['label'].value_counts())

# Setup for Google Drive remote storage (Bonus)
def setup_gdrive_remote():
    print("\n\n==== Setting up Google Drive remote storage ====")
    # Add Google Drive as remote storage
    run_command('dvc remote add -d myremote gdrive://1ABCxyz123') # Replace with your Google Drive folder ID
    
    # Modify .dvc/config to use Google Drive
    run_command('git add .dvc/config')
    run_command('git commit -m "Configure remote storage with Google Drive"')
    
    # Push data to Google Drive
    run_command('dvc push')
    
    print("\nData pushed to Google Drive")

# Main execution flow
save_raw_data()
init_dvc()
create_first_version()
create_second_version()
checkout_first_version()
checkout_second_version()

# Uncomment to use Google Drive remote storage (Bonus)
# setup_gdrive_remote()
