from flask import *
import pandas as pd
import requests, json
import time
import numpy as np

from tinydb import TinyDB, Query

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

TOP = 20

TRACK_TABLE = TinyDB('/home/sergey.miller/research/data/prod.db').table('tracks')


@app.route('/predict', methods=['GET'])
def predict():
    x = float(request.args.get("x"))
    y = float(request.args.get("y"))
    top = int(request.args.get("top"))
    if top is None:
        top = TOP
    resp = TRACK_TABLE.all()
    resp = [{'mac': x['mac'], 'pred': list(map(float,x['pred'].split(',')))} for x in resp]
    sorted_points = sorted(resp, key=lambda P: (float(P['pred'][0]) - x) ** 2 + (float(P['pred'][1]) - y) ** 2)
    sorted_points = sorted_points[:top]
    res = []
    for _it in sorted_points:
        res.append(make_point([float(_it['pred'][0]),float(_it['pred'][1])], floor=int(np.clip(float(_it['pred'][2]), a_min=0, a_max=3)), mac=_it['mac']))
    return json.dumps(res)



@app.route('/predict_around', methods=['GET'])
def predict_around():
    res = json.dumps([make_point(p, str(i)) for i, p in enumerate(STATIC_POINTS)])
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
