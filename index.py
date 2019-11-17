from flask import *
import pandas as pd
import requests, json
import time
import numpy as np

from tinydb import TinyDB, Query

from collections import defaultdict

import kalman

app = Flask(__name__)

def make_point(p, floor=1, mac=''):
    return {'lat': p[0], 'lon': p[1], 'floor': floor, 'id': mac}

STATIC_POINTS = np.array([
    [60.185374, 24.824871],
    [60.185430, 24.824495],
    [60.185585, 24.824205],
    [60.185525, 24.824604],
    [60.185469, 24.824695],
    [60.185837, 24.825007]
])

# =60.185469&lon=24.824695

TOP = 20

TRACK_TABLE = TinyDB('/home/sergey.miller/research/data/prod.db').table('tracks')


CARS = dict()


def smooth(mac, p, data):
    mu, for_ts = data
    car = kalman.Car(mu[:3], mu[3:6], for_ts)
    car.move(time.time() - for_ts)
    return list(car._state)


@app.route('/predict', methods=['GET'])
def predict():
    x = float(request.args.get("lat"))
    y = float(request.args.get("lon"))
    ub = int(request.args.get("ub"))
    lb = int(request.args.get("lb"))
    if ub is None or lb is None:
        ub = TOP
        lb = 0
    resp = TRACK_TABLE.all()
    resp = [{'kalman': [list(map(float,x['kalman_mu'].split(','))), x['for_ts']], 'mac': x['mac'], 'pred': list(map(float,x['pred'].split(',')))} for x in resp]
    resp = [{'pred': smooth(x['mac'], x['pred'], x['kalman']), 'mac': x['mac']} for x in resp]
    sorted_points = sorted(resp, key=lambda P: (float(P['pred'][0]) - x) ** 2 + (float(P['pred'][1]) - y) ** 2)
    sorted_points = sorted_points[lb:ub]
    res = []
    for _it in sorted_points:
        res.append(make_point([float(_it['pred'][0]),float(_it['pred'][1])], floor=int(np.clip(float(_it['pred'][2]), a_min=0, a_max=3)), mac=_it['mac']))
    return json.dumps(res)



@app.route('/predict_around', methods=['GET'])
def predict_around():
    res = json.dumps([make_point(p, 1, str(i)) for i, p in enumerate(STATIC_POINTS)])
    return res

@app.route('/linear', methods=['GET'])
def linear_movement():
    ts = time.time()
    T = 30
    H = int(T // 2)
    side = int(ts) % T >= H
    s,f = STATIC_POINTS[0], STATIC_POINTS[1]
    if side:
        s,f = f,s
    alpha = (ts - int(int(ts) // H) * H) / H
    cur = s * alpha + f * ( 1 - alpha)
    return json.dumps([make_point(cur, floor=1,mac=str(0))])

if __name__=='__main__':
	app.run(host='0.0.0.0')
