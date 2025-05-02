![](sp2025-movie-idea/imgs/banner.png)

# EXTRA EXTRA READ ALL ABOUT IT!

[Movie-Plot-Generator](https://huggingface.co/spaces/tdoucet2/movie-plot-generator) is a fun exploratory project that is designed for movie fans, screenwriters, creatives, and anyone curious about the power of fine-tuning a language model.  

## Goals
  
The project developers set out with a few very specific goals before attempting this feat.

 - Find the highest quality movie plots for testing, training, and validation.
 - Fine-tune a top-performing BART-based summarization model
 - Apply state-of-the-art evaluation techniques post fine-tuning
 - Build an aesthetically pleasing and functional GUI for MVP rollout

 ## Methods

 ### Data Exploration
Finding the movie plots for model training was the **MOST** enjoyable aspect of this project (everything was fun, but this was the best part).  

[The Movie Database](https://www.themoviedb.org) has a sweet API where you can pull movie titles and overviews from their most popular, top-rated, and now-playing pages. We used our **collect_movie_descriptions():** function (found in **data_gathering.ipynb**) to run for 40 minutes to collect over 16,000 titles!

Here's a fun word cloud we created from our data exploratory analysis using this dataset: 

![](sp2025-movie-idea/imgs/wordcloud.png)  

The bulk of our training data came from [Kaggle: Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). The description for this dataset: The dataset contains descriptions of 34,886 movies from around the world. The size and quality of this data was hard to beat and we highly recommend this source!

### Fine-Tuning

We went with [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) due to it being one of the strongest out-of-the-box models for abstractive summarization. Since our project centers around generating creative plot summaries from imaginative titles, BART gave us the flexibility to fine-tune on structured prompt/plot pairs.

We wrote a custom **preprocess()** function that handles both the tokenization and padding for the input titles and target plots. A cool addition we threw in was a custom **LogCallback** class that tracks training and evaluation logs at each step. On top of that, we added early stopping to avoid overfitting and enabled TensorBoard logging for more granular insights. Lastly, we calculated perplexity. You can never have enough metrics!

Here is our model working and learning - just as we are!

![](sp2025-movie-idea/imgs/training.png)

### Evaluation

Once we had the fine-tuned model saved, it was time to dive into some grid search parameter tuning and cosine similarity testing to see how our outputs stacked up against the original plots. 

We ran a series of lightweight generation trials using a small validation sample. With limited resources, we sampled 50 examples and explored a grid of decoding parameters: temperature, top-k, top-p, length, and n-gram constraints. The goal was to see which combination of parameters yielded the most coherent and creative summaries. Each configuration was tested and scored using ROUGE-L F1, and we logged the top-performing setups while using early stopping to avoid wasting cycles. 

With our best-performing config extracted from the ROUGE testing, we ran the model across a genre-balanced sample and used cosine similarity to compare generated plots to their originals. This gave us a sense of how close the summaries were in meaning. 

Here was the result of the experiment:

![](sp2025-movie-idea/imgs/cosine.png)

### DEMO

Our fine-tuned model is hosted on hugging face for public use. We created a GUI using gradio and Hugging Face's Spaces hosting space. Here is the [link](https://huggingface.co/spaces/tdoucet2/movie-plot-generator) to the demo!




## Conclusion

The project was about blending passion with hands-on learning to build something our team was genuinely proud to submit. From data collection to model training to evaluation, we explored what it takes to turn a movie title into a compelling plot - one prompt at a time. 