"""Sample client for track 2 of IPIN2022 competition.

Example calls:
 Test on a randomly selected trial
   $ python3 track2_client.py
 Test on a specific trial
   $ python3 track2_client.py trial_12_0
"""
import json
import requests
import random
import numpy as np
from PIL import Image
from io import BytesIO
# from datetime import datetime, timezone
from time import perf_counter
from pprint import pprint
from sys import argv
from requests.adapters import HTTPAdapter, Retry
from visual_odometry_for_trail import camloc, VisualOdometry, camloc2

import cv2

# Reference ----------------------------------------------------------
# TRIAL_IDS = ("trial_11_0","trial_12_0","trial_13_0","trial_14_0","trial_15_0")
# TRIAL_STATES = ("UNINITIALIZED","NONSTARTED","RUNNING","FINISHED","STOPPED","TIMEOUT")
# TRIAL_CMDS = ("INIT","START","STOP","NEXTDATA")


# Configuration ------------------------------------------------------
USER_ID = "xxx"
USER_PW = "xxx"
TRIAL_ID = None
LOGIN_IP = "xxx"
LOGIN_PORT = "xxx"

URLBase = f'http://{LOGIN_IP}:{LOGIN_PORT}/api'  # Login URL
login_headers = {'Content-Type': 'application/json; charset=utf-8'} 
trial_header = {'Content-Type': 'application/json'}

trl_session = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
                status_forcelist=[500, 502, 503, 504])
trl_session.mount('http://', HTTPAdapter(max_retries=retries))

# Localization -------------------------------------------------------
def localize(observation):
    """Estimate a pose from the given observation.

    TODO: Replace with own localization implementation.

    Parameters
    ----------
    observation : numpy array
        An input image (HxWxC: 640x480x3)

    Returns
    -------
    dict
        The expected dictionary items are:
        lon : float
           Estimated longitude.
        lat : float 
           Estimated latitude.
        floor : int
           Estimated floor.
        proc_time : float
           Processing time in seconds.
        q_w, q_x, q_y, q_z : float
           Orientation quaternion.  May return the default unit quaternion 
           as the orientation is expected, but will not be used for scoring.
    """
    ts0 = perf_counter()
    cv2.imshow("Image", observation[:, :, ::-1])
    cv2.waitKey(5)
    # print("Localization on ", observation.shape)
    lon = 127.370282 + random.randrange(-10, 10, 1) * 1e-4
    lat = 36.38365   + random.randrange(-10, 10, 1) * 1e-4
    floor = 3
    proc_time = perf_counter() - ts0
    q_w, q_x, q_y, q_z = 1.0, 0.0, 0.0, 0.0
    return {"lon": lon, "lat": lat, "floor": floor, "proc_time": proc_time, "q_w": q_w, "q_x": q_x, "q_y": q_y, "q_z": q_z}


# Server communication -----------------------------------------------
def decode_image(resp):
    """Decode image_id and image_data from server response.

    Parameters
    ----------
    resp : dict
        Response received from the test server with the following items
                'image_id':  str
                'image_data': str

    Returns
    -------
    tuple : (str, numpy array)
        A tuple of image id and image data (HxWxC: 640x480x3).

    """
    image_id = resp.get("image_id")
    im = Image.open(BytesIO(resp.get("image_data").encode('latin1')))
    image_data = np.array(im)
    return image_id, image_data


def init_trial(trial_url, user_id, session_id, trial_id=None):
    """Initialize a trial.

    Parameters
    ----------
    trial_url : str
        URL for trial requests.
    user_id : str
        User id to send to the server.
    session_id : str
        Session id to send to the server.
    trial_id : str, optional
        Trial id, if not provided, the trial will be selected randomly on the server-side.

    Returns
    --------
    dictionary
        The server response as a dictionary. Possible items include:
        'user_id' : str
            User id, should be the same as the input parameter.
        'session_id' : str
            Session id, should be the same as the input parameter.
        'trial_id' : str
            Trial id, should be the same as the input parameter if provided or randomly chosen on the server side.
        'trial_state' : str
            Trial state should be NONSTARTED
        'error' : str
            If an error occurred at the server side, this item will contain some explanation.
    """
    req = {"user_id": user_id,
           "session_id": session_id,
           "trial_cmd": "INIT"}
    if trial_id is not None:
        req['trial_id'] =  trial_id

    sreq = json.dumps(req) 
    resp = requests.post(trial_url, json=sreq, headers=trial_header)
    try:
        respj = resp.json()
    except Exception as e:
        print("init_trial, json error", e)
        respj = {'error': 'init_trial response json error'}

    return respj


def start_trial(trial_url, user_id, session_id, trial_id):
    """Start a trial.

    Parameters
    ----------
    trial_url : str
        URL for trial requests.
    user_id : str
        User id to send to the server.
    session_id : str
        Session id to send to the server.
    trial_id : str
        Trial id, should be the one that was returned from init_trial.

    Returns
    --------
    dictionary
        The server response as a dictionary. Possible items include:
        'user_id' : str
            User id, should be the same as the input parameter.
        'session_id' : str
            Session id, should be the same as the input parameter.
        'trial_id' : str
            Trial id, should be the same as the input parameter.
        'trial_state' : str
            Trial state should be RUNNING
        'image_id':  str
            First image id.
        'image_data': str
            First image data.
        'error' : str
            If an error occurred at the server side, this item will contain some explanation.
    """
    req = {"user_id": user_id,
           "session_id": session_id,
           "trial_id": trial_id,
           "trial_cmd": "START"}

    sreq = json.dumps(req) 
    resp = requests.post(trial_url, json=sreq, headers=trial_header)
    try:
        respj = resp.json()
    except Exception as e:
        print("start_trial, json error", e)
        respj = {'error': 'start_trial response json error'}

    return respj
    

def next_data(trial_url, user_id, session_id, trial_id, estimation=None):
    """Request next data sample (next image).

    If estimation is not provided, a request to stop the trial will be send.

    Parameters
    ----------
    trial_url : str
        URL for trial requests.
    user_id : str
        User id to send to the server.
    session_id : str
        Session id to send to the server.
    trial_id : str
        Trial id, should be the one that was returned from init_trial.
    estimation : dict | None
        The estimated pose as returned by localize(observation).

    Returns
    --------
    dictionary
        The server response as a dictionary, containing the next observation if available.
        'user_id' : str
            User id, should be the same as the input parameter.
        'session_id' : str
            Session id, should be the same as the input parameter.
        'trial_id' : str
            Trial id, should be the same as the input parameter if provided or randomly chosen on the server side.
        'trial_state' : str
            Trial state should be RUNNING if more images are available or FINISHED otherwise.
            If no estimation was provided, the request was sent to stop the trial and this value should be STOPPED.
        'image_id':  str
            Next image id, if available.
        'image_data': str
            Next image data, if available.
        'error' : str
            If an error occurred at the server side, this item will contain some explanation.
    """
    trial_cmd = "NEXTDATA" if estimation is not None else "STOP"

    req = {"user_id": user_id,
           "session_id": session_id,
           "trial_id": trial_id,
           "estimation": estimation,
           "trial_cmd": trial_cmd}

    sreq = json.dumps(req) 
    # resp = requests.post(trial_url, json=sreq, headers=trial_header)
    resp = trl_session.post(trial_url, json=sreq, headers=trial_header)
    try:
        respj = resp.json()
    except Exception as e:
        print("next_data, json error", e)
        respj = {'error': 'next_data response json error'}

    return respj


def login(user_id, pw):
    data = {'user_id': user_id, 'pw': USER_PW}
    res = requests.post(URLBase+'/login_track2', json=data, headers=login_headers)
    return res


def logout(user_id, session_id):
    data = {'user_id': user_id, 'session_id': session_id} 
    res = requests.post(URLBase+'/logout_track2', json=data, headers=login_headers)
    return res


def main(trial_id=None):
    user_id = USER_ID
    # Login
    res = login(user_id, USER_PW)
    print("login res", res)
    session_id = res.json().get('session_id')
    if session_id is None or session_id == '':
        print(f"Login failed")
        exit()

    trial_url = res.json().get('url')
    if trial_url is None:
        print(f"trial_url missing from login response")
        logout(user_id, session_id)
        exit()

    # Login successful
    # Initialize trial
    print(f"init_trial {user_id} {session_id} {trial_id}")
    resp = init_trial(trial_url, user_id, session_id, trial_id=trial_id)
    err = resp.get("error", None)
    if err is not None:
        print(f"init_trial error: {err}")
        logout(user_id, session_id)
        exit()
    trial_id = resp.get("trial_id")
    trial_state = resp.get("trial_state")
    phone_model = resp.get("phone_model")
    print(f" response: {user_id} {session_id} {trial_id} {trial_state} {phone_model}")

    # Start trial
    print(f"start_trial {user_id} {session_id} {trial_id}")
    resp = start_trial(trial_url, user_id, session_id, trial_id)
    err = resp.get("error", None)
    if err is not None:
        print(f"start_trial error: {err}")
        logout(user_id, session_id)
        exit()

    trial_state = resp.get("trial_state")
    print(f"  response: {user_id} {session_id} {trial_id} {trial_state}")

    # Loop over trial images
    finished = (resp.get("trial_state") in ["FINISHED", "STOPPED", None])
    n = 0

    print('System Start!')
    method = 'superpoint'
    matcher = 'superglue'
    vo = VisualOdometry(method=method, matcher=matcher)
    align_transformation = np.eye(4)
    # qua_pose = []
    # initial_longitude, initial_latitude = [0.0], [0.0]
    global_lat, global_lon, global_alt = [36.383650], [127.370282], [0]
    floor = [0]
    cur_pose = np.array([
        [1., 0., 0., 0.0],
        [0., 1. , 0., 0.0],
        [0., 0., 1., 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size + 200, 3), dtype=np.uint8)

    while not finished:
        img_id, img_data = decode_image(resp)
        print(f"{trial_state} {n:3d}: {img_id} {img_data.shape}")
        if img_data is None:
            print(f"loop_trial ({n:2d}) error: image_data unavailable.")
            estimation = None
        else:
            # TODO: Modify the following call. 
            # estimation = localize(img_data)
            # vo.images.append(cv2.cvtColor(np.asarray(img_data), cv2.COLOR_RGB2GRAY))
            estimation = camloc2(vo=vo, image_id=n, image_data=img_data, floor=floor,
                                cur_pose=cur_pose, traj_img=traj_img, align_transformation=align_transformation,
                                 global_lat=global_lat, global_lon=global_lon, global_alt=global_alt)

        resp = next_data(trial_url, user_id, session_id, trial_id, estimation)
        err = resp.get("error", None)
        if err is not None:
            print(f"next_data error: {err}")
            logout(user_id, session_id)
            exit()

        n += 1
        finished = (resp.get("trial_state") in ["FINISHED", "STOPPED", None])

    # Trial finished/stopped
    trial_state = resp.get("trial_state")
    print(f"  response: {user_id} {session_id} {trial_id} {trial_state}")
    logout(user_id, session_id)

    # Print out the test results if any.
    tresults = resp.get("trial_score")
    # tresults['err_f']
    #   is the floor error, computed on all samples.
    # tresults['err_p']
    #   is the Euclidean horizontal distance between estimation and GT,
    #   computed on all samples.
    # tresults['err75']
    #   is the 3rd quartile of the point errors (err_f + err_p),
    #   computed on the reference points only 
    print("Test results:")
    pprint(tresults, compact=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(argv) >= 2:
        TRIAL_ID = argv[1]
    
    main(trial_id=TRIAL_ID)



