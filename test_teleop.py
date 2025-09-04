import requests

url_left = "http://100.72.16.108:5000/"
url_right = "http://100.72.16.108:5001/"

curr_action = {"right_arm": {"pose": [0., 0., 0., 0., .0, 0.],
                                              "gripper": [0.],
                                              "activate_button": [0.]},
                                "left_arm": {"pose": [0., 0., 0., 0., .0, 0.],
                                              "gripper": [0.],
                                              "activate_button": [0.]}
            }



while 1:
    ps_left = requests.post(url_left + "get_pico_action").json()
    ps_right = requests.post(url_right + "get_pico_action").json()
    curr_action["right_arm"]["pose"] = ps_right["pose"]
    curr_action["right_arm"]["gripper"] = ps_right["gripper"]
    curr_action["left_arm"]["pose"] = ps_left["pose"]
    curr_action["left_arm"]["gripper"] = ps_left["gripper"]
    print(curr_action)