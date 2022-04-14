import concurrent.futures
import time

from pip import main
import requests
import json


def main_api_call():
    main_url = 'https://formulae.brew.sh/api/formula.json'
    base_url = 'https://formulae.brew.sh/api/formula/'
    try:
        with requests.get(main_url) as response:
            if response:
                print(response.json())
    except requests.HTTPError:
        return response.status_code


def any_call(url):
    response = requests.get(url)
    response_json = response.json()
    with open('res.json', 'w') as res:
        res.write(response_json)
        print(response.status_code)


def recursive_api_calls():
    main_url = 'https://formulae.brew.sh.api/formula.json'
    base_url = 'https://formulae.brew.sh.api/formula/'
    try:
        with requests.get(main_url) as response:
            return response.json()
    except requests.HTTPError:
        return response.status_code


def recursive_api_calls():
    main_url = 'https://formulae.brew.sh/api/formula.json'
    main_response = requests.get(main_url)
    for item in main_response:
        name = item['name']
        recursive_url = f'https://formulae.brew.sh/api/formula/{name}.json'
        response_call = any_call()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(any_call, recursive_url)

        # recursive_call_response = any_call(recursive_url)
        # res_str = json.dumps(recursive_call_response, indent=2)
        # print(res_str)


def helper_to_get_api_objects():
    url = []
    main_url = 'https://formulae.brew.sh/api/formula.json'
    main_response = requests.get(main_url)
    main_res_json = main_response.json()
    for item in main_res_json:
        name = item['name']
        urls = f'https://formulae.brew.sh/api/formula/{name}.json'
        url.append(urls)
    return url


def api_calls():
    urls = helper_to_get_api_objects()
    for url in urls:
        recursive_calls_solution(url)


def recursive_calls_solution(url):
    response = requests.get(url)
    response_json = response.json()
    res_str = json.dumps(response_json, indent=2)
    print(res_str)
    with open('res_asy.json', 'w') as log:
        log.write(res_str)
        print(f'{log} was log')


if __name__ == '__main__':
    t1 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as exec:
        url = helper_to_get_api_objects()
        exec.map(recursive_calls_solution, url)

    t2 = time.perf_counter()

    print(f'Finished in {t2 - t1} seconds')
