import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
num_tweets = 100

np.random.seed(0)
tweets = ["I love this product!",
          "The customer service is terrible.",
          "Great experience with this brand!",
          "I'm so disappointed with the quality.",
          "Amazing features!",
          "Worst purchase ever.",
          "The new update is fantastic!",
          "Poor packaging, but the product works well.",
          "So happy with my purchase!",
          "This company never fails to impress me."]

sentiment_scores = np.random.uniform(-1, 1, num_tweets)

data = {'Tweet': np.random.choice(tweets, num_tweets),
        'Sentiment Score': sentiment_scores}
df = pd.DataFrame(data)

df.to_csv('sample_twitter_data.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Sentiment Score'], marker='o', linestyle='-')
plt.title('Sentiment Trends in Sample Tweets')
plt.xlabel('Tweet Index')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()
