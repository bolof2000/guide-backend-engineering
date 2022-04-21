from faker import Faker
from faker import Factory
fake_data1 = Faker()
fake_data = Factory.create()
import json
import requests

def generate_api_payload():
    name = fake_data.first_name()
    job = fake_data.last_name()
    payload = {
        "name":name,
        "job":job
    }

    return payload


def test_create_new_user():
    payload = generate_api_payload()

    response = requests.post("https://reqres.in/api/users", payload)
    response_json = response.json()
    json_str = json.dumps(response_json,indent=2)
    print(json_str)
    assert response.status_code == 201


def test_create_new_user2():
    json_str = {
    "name": "volof",
     "job": "enginer"
    }

    response = requests.post("https://reqres.in/api/users", json_str)
    response_json = response.json()
   # json_str = json.dumps(response_json, indent=2)
    json_str_von = json.dumps(json_str,indent=2)
    print(json_str_von)
    assert response.status_code == 201