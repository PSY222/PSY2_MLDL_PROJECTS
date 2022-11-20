# Competition Overview & Background

This competition was held by Korea data science competition website 'DACON'.
Jeju island is one of the Korea's best touristy place with beautiful natural environments and things to enjoy! Popularity of Jeju island is steadily increasing that the total population of Jeju is expected to increase by millions in few years.

So, Jeju government hosted competition that predicts the future traffic volume. Given data covers traffic data (start location, date, road type...etc). 

# My Trials and Errors
I began approach with step-by-step EDA of each data features. I sorted out unnecessary columns, null values and way to deal with string values with hundreds of unique values. Then, I tried multiple machine learning regression models with simple tensorflow model. Although I aimed for stacking regressor model, my colab experiment environment couldn't handle large size data that it even crashed when I tried PCA transformed data. 

However, this competition was meaningful for getting used to data preprocessing process and being much familiar with regression models using hyperparameter tuning. 
