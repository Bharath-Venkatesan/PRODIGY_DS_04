import pandas as pd
import matplotlib.pyplot as plt

# Load the training dataset
train_df = pd.read_csv("E:/ProdigyInfoTech_DataScience_Tasks/twitter_training.csv")

# Load the validation dataset
validation_df = pd.read_csv("E:/ProdigyInfoTech_DataScience_Tasks/twitter_validation.csv")

# Assuming 'date' and 'sentiment_score' columns exist in the datasets
# Convert 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
validation_df['date'] = pd.to_datetime(validation_df['date'])

# Plot sentiment trends over time for training data
plt.figure(figsize=(10, 6))
plt.plot(train_df['date'], train_df['sentiment_score'], marker='o', linestyle='-', label='Training Data')
plt.title('Sentiment Trends Over Time (Training Data)')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.legend()
plt.show()

# Plot sentiment trends over time for validation data
plt.figure(figsize=(10, 6))
plt.plot(validation_df['date'], validation_df['sentiment_score'], marker='o', linestyle='-', color='orange', label='Validation Data')
plt.title('Sentiment Trends Over Time (Validation Data)')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.legend()
plt.show()
