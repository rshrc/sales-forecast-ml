# Sales Forecast using Machine Learning

### A web application to predict the sales of a newly launched product
#### Libraries Used : scikit-learn Web Framework User : Django
##### Using the pickle library to load and use pre trained ml models.

### How does it look like when you run it?
#### Main Page
![MainPage](https://i.imgur.com/mjIUd6T.png)

#### Product Detail Input Page
![ProductInput](https://i.imgur.com/OgrFdvj.png)

#### Prediction Output Page
![PredictionPage](https://i.imgur.com/ywBj5KY.png)


### How to run sales_predit_ml on your Linux/Unix System ?

##### Clone the repository and get inside sales_forecast_ml
```
git clone https://github.com/rshrc/sales_forecast_ml && cd sales_predict_ml
```

##### Installing required Python3 libraries
```
sudo pip3 install -r requirements.txt
```

##### Make Migrations
```
python3 manage.py makemigrations predict && python3 manage.py migrate
```

##### Loading the server_predictor.py for later use
```
python3 ml_core/ml_process/server_predictor.py
```

##### Running the Server
```
python3 manage.py runserver
```

You should now be able to access the sales_predict_ml web app in localhost:8000 in your browser

### How to run sales_predit_ml on your Windows System ?
Install Linux or buy a Mac and revisit https://github.com/rshrc/sales_forecast_ml/README.md
