# Competition Overview & Background

This competition was held by Korea data science competition website 'DACON'.
Jeju island is one of the Korea's best touristy place with beautiful natural environments and things to enjoy! Popularity of Jeju island is steadily increasing that the total population of Jeju is expected to increase by millions in few years. So, Jeju government hosted the competition aiming for the stable transport environment development based on the future traffic volume prediction. <br>

![image](https://user-images.githubusercontent.com/86555104/209466331-c4a86ca9-d185-41c1-88d2-28ef16e6e9aa.png)

# Data Overview
Data size : (4701217, 23)  <br>
Columns : id, base_date, day_of_week, base_hour, lane_count, road_rating, road_name, multi_linked, connect_code, maximum_speed_limit, vehicle_restricted, weight_restricted, height_restricted, road_type, start_node_name, start_latitude, start_longitude, start_turn_restricted, end_node_name, end_latitude, end_longitude, end_turn_restricted, target <br>
**target**: predicting the average speed(km) of the vehicle


# Approach
![image](https://user-images.githubusercontent.com/86555104/209467374-95209c0f-a5a6-4234-be72-5a8d901354db.png)


During the EDA process, I discovered that there were particular features with a biased distribution(one value had over 80-99% of the total data) and duplicated information(identical data in different feature). Based on the findings, those features were dropped before the training. 'start_node_name''end_node_name' had 487 unique values which made it hard to deal with OneHotEncoder. Starting/ending point of the vehicle was a valuable information dropping the column couldn't be the option as well. To deal with this problem, I categorized 487 features into 9 different categories such as living, entrance, and village name making it easier to apply OneHotEncoder.

Then, I tried multiple machine learning regression models with simple tensorflow model. Among 6 different models, Ridge < GBR > LGBM model showed good performance on the cross validation with KFold and RMSE measure.

Although I aimed for stacking regressor model, my colab experiment environment couldn't handle the large size data that it crashed even when I tried PCA to minimize the dimension.

![image](https://user-images.githubusercontent.com/86555104/209467479-8cf21832-9b35-4497-b238-a3138d889f81.png)
