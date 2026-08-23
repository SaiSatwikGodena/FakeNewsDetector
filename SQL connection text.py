import mysql.connector
from db_config import DB_CONFIG
import db_config

print("Using config from:", db_config.__file__)
print("Password length:", len(DB_CONFIG["password"]))

conn = mysql.connector.connect(**DB_CONFIG)

print("CONNECTED SUCCESSFULLY!")
conn.close()