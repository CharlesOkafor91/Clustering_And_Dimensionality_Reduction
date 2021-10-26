# Clustering_And_Dimensionality_Reduction

Peer-graded Assignment: Course Project
UnSupervised Machine Learning:
Classification
Main objective: For this project, I will be using various unsupervised machine learning models
for dimensionality reduction to reduce the number of features in my dataset using the kaggle
competition dataset here. The purpose will be to reduce the final features that will be used in model
prediction of the Ames housing with the advanced complex features provided. This will answer the
question of what features are important enough to contribute to the prediction of the house prices..
Brief description of the data set: The dataset was extracted from the Ames data Housing on
Kaggle. This presents a more advanced set of features in predicting house prices. If you ask a home
owner/buyer to describe what they prefer in their dream houses, they may likely not only mention
things like heights of the basement, etc. They are more likely to mention proximity to some certain
areas, amenities available, number of bedrooms, etc.
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa,
we are to predict the final price of each house.
Column Description
SalePrice The property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass The building class
MSZoning The general zoning classification
LotFrontage Linear feet of street connected to property
LotArea Lot size in square feet
Street Type of road access
Alley Type of alley access
LotShape General shape of property
LandContour Flatness of the property
Utilities Type of utilities available
LotConfig Lot configuration
LandSlope Slope of property
Neighborhood Physical locations within Ames city limits
Condition1 Proximity to main road or railroad
Condition2 Proximity to main road or railroad (if a second is present)
BldgType Type of dwelling
HouseStyle Style of dwelling
OverallQual Overall material and finish quality
OverallCond Overall condition rating
YearBuilt Original construction date
YearRemodAd
d
Remodel date
RoofStyle Type of roof
RoofMatl Roof material
Exterior1st Exterior covering on house
Exterior2nd Exterior covering on house (if more than one material)
MasVnrType Masonry veneer type
MasVnrArea Masonry veneer area in square feet
ExterQual Exterior material quality
ExterCond Present condition of the material on the exterior
Foundation Type of foundation
BsmtQual Height of the basement
BsmtCond General condition of the basement
BsmtExposur
e
Walkout or garden level basement walls
BsmtFinType1 Quality of basement finished area
BsmtFinSF1 Type 1 finished square feet
BsmtFinType2 Quality of second finished area (if present)
BsmtFinSF2 Type 2 finished square feet
BsmtUnfSF Unfinished square feet of basement area
TotalBsmtSF Total square feet of basement area
Heating Type of heating
HeatingQC Heating quality and condition
CentralAir Central air conditioning
Electrical Electrical system
1stFlrSF First Floor square feet
2ndFlrSF Second floor square feet
LowQualFinS
F
Low quality finished square feet (all floors)
GrLivArea Above grade (ground) living area square feet
BsmtFullBath Basement full bathrooms
BsmtHalfBath Basement half bathrooms
FullBath Full bathrooms above grade
HalfBath Half baths above grade
Bedroom Number of bedrooms above basement level
Kitchen Number of kitchens
KitchenQual Kitchen quality
TotRmsAbvGr
d
Total rooms above grade (does not include bathrooms)
Functional Home functionality rating
Fireplaces Number of fireplaces
FireplaceQu Fireplace quality
GarageType Garage location
GarageYrBlt Year garage was built
GarageFinish Interior finish of the garage
GarageCars Size of garage in car capacity
GarageArea Size of garage in square feet
GarageQual Garage quality
GarageCond Garage condition
PavedDrive Paved driveway
WoodDeckSF Wood deck area in square feet
OpenPorchSF Open porch area in square feet
EnclosedPorc
h
Enclosed porch area in square feet
3SsnPorch Three season porch area in square feet
ScreenPorch Screen porch area in square feet
PoolArea Pool area in square feet
PoolQC Pool quality
Fence Fence quality
MiscFeature Miscellaneous feature not covered in other categories
MiscVal $Value of miscellaneous feature
MoSold Month Sold
YrSold Year Sold
SaleType Type of sale
SaleCondition Condition of sale
Brief summary of data exploration and Feature Engineering: First I changed the
datatype of the column MoSold (which is the month the property was sold) from int to dtype category.
Then I calculated Age for all the columns with year values and replaced the year values with Age
instead. I then replaced all the missing values for both categorical and numerical data with their
respective values. I then took the log transformation of numerical columns that were skewed. Finally
I scaled the data using MinMaxScaler and encoded the categorical data with LabelEncoder and
OneHotEncoder.
Below shows some code snippet for the data exploration:
Summary of training at least three variations of the unsupervised models:
1. PCA: We used PCA for dimensionality reduction. We were able to realise that the total
number of components needed in our dataset was about 95 out of the 311 total features
initially generated. Here is our metrics after we used ElasticCV to predict our sales price:
2.
3. K MEANS: Next we used Kmeans to cluster our data to their respective SaleCondition
values. Here is how we did compared to the original SaleCondition column:
4. Agglomerative Clustering: Lastly we used the Agglomerative Clustering to cluster the
same SaleCondition values. Here is how we did compared to the original SaleCondition
column:
Recommendation: Based on our findings, I will recommend using DBSCAN to cluster our data
for more accuracy.
Summary Key Findings and Insights: The PCA did a great job in reducing the dimensions
of data which helped in reducing the MSE of our regression. K means and Agglomerative clustering
was also used to cluster our datasets using the Sales Conditions as the clusters..
My notebook can be found here
