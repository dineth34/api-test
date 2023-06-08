from flask import Flask
import urllib.request
import os

app = Flask(__name__)

stores = [{"name": "My Store", "items": [{"name": "my item", "price": 15.99}]}]

def get_file_details():
    current_location = os.getcwd()  # Get the current working directory
    
    file_details = []  # List to store file details
    
    for root, dirs, files in os.walk(current_location):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            file_details.append({
                'file_name': file,
                'file_path': file_path,
                'file_size': file_size
            })
    
    return file_details

@app.route('/store', methods=['GET'])
def get_stores():
    urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg', 'yolov4.cfg')
    urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights', 'yolov4.weights')
    return {"stores": stores}

@app.route('/test', methods=['GET'])
def test():
    # Usage
    files = get_file_details()

    # Print file details
    for file in files:
        print("File Name:", file['file_name'])
        print("File Path:", file['file_path'])
        print("File Size (bytes):", file['file_size'])
        print("-" * 30)
    return "File view option successful!"

if __name__ == '__main__':
    app.run(debug=false, host='0.0.0.0')