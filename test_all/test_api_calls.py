import requests 
import json 
import pytest
import os


@pytest.fixture
def load_test_data_fixture():
    global file 
    file = open("create_user.json",'r')
    
def test_create_new_user(load_test_data_fixture):
    json_str = file.read()
    payload = json.loads(json_str)
    response = requests.post("https://reqres.in/api/users",payload)
    assert response.status_code == 201
    