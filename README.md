# Verify Your News Article
## Predicts News article given as a fake or a real news!
### for more accuracy, there must be more data in the Fake.CSV and the True.CSV

## Steps to setup the code:
### Step1
Setup your SQL Database by running:
```mysql -u root -p < database_setup.sql```
### Step2
Download the python dependencies:
```pip install -r requirements.txt```
### Step3
In the db_config.py file
> Change "your_mysql_password_here" to <your ACTUAL password>

## Training your model
Once you have your Dataset in the from of CSVs
run: 
```python load_data.py --fake Fake.csv --real True.csv```

## Running the model
simply run : 
```python fake_news_system.py```
# Demo of how the run looks like!!
<img width="1387" height="823" alt="image" src="https://github.com/user-attachments/assets/aaf33e1f-370b-4425-a381-bd39ed34fbfa" />
# MatplotLib Graph basedon the training model
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/7ee5469a-2746-44ff-ba2b-80cc2bdcfc9b" />

