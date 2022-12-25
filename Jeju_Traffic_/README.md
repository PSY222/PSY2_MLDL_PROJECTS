# Competition Overview & Background

This competition was held by Korea data science competition website 'DACON'.
Jeju island is one of the Korea's best touristy place with beautiful natural environments and things to enjoy! Popularity of Jeju island is steadily increasing that the total population of Jeju is expected to increase by millions in few years. So, Jeju government hosted the competition aiming for the stable transport environment development based on the future traffic volume prediction. 

![image](https://user-images.githubusercontent.com/86555104/209466331-c4a86ca9-d185-41c1-88d2-28ef16e6e9aa.png)

# Data Overview
Data size : 4,701,217 rows <br>
Columns : id, base_date, day_of_week, base_hour, lane_count, road_rating, road_name, multi_linked, connect_code, maximum_speed_limit, vehicle_restricted, weight_restricted, height_restricted, road_type, start_node_name, start_latitude, start_longitude, start_turn_restricted, end_node_name, end_latitude, end_longitude, end_turn_restricted, target
target: predicting the average speed(km) of the vehicle


# Approach
I began approach with step-by-step EDA of each data features. I sorted out unnecessary columns, null values and way to deal with string values with hundreds of unique values. Then, I tried multiple machine learning regression models with simple tensorflow model. Although I aimed for stacking regressor model, my colab experiment environment couldn't handle large size data that it even crashed when I tried PCA transformed data. 

# Key Takeaways
This competition was meaningful for getting used to data preprocessing process and being much familiar with regression models using hyperparameter tuning. 
