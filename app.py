from flask import Flask
import urllib.request

app = Flask(__name__)

stores = [{"name": "My Store", "items": [{"name": "my item", "price": 15.99}]}]

@app.route('/store', methods=['GET'])
def get_stores():
    urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg', 'yolov4.cfg')
    urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights', 'yolov4.weights')
    return {"stores": stores}

if __name__ == '__main__':
    app.run(debug=false, host='0.0.0.0')