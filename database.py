import os
from deta import Deta
from dotenv import load_dotenv

load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")
deta = Deta(DETA_KEY)

db = deta.Base("potential_employees")

def insert_data(key, name, gender, picture, experience, hobbies, resume):
    return db.put({"key": key, "name": name, "gender": gender, "picture": picture, "experience": experience, "hobbies": hobbies, "resume": resume})

def fetch_all_data():
    res = db.fetch()
    return res.items
