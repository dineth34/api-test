from flask import Flask

app = Flask(__name__)

stores = [{"name": "My Store", "items": [{"name": "my item", "price": 15.99}]}]

@app.route('/store', methods=['GET'])
def get_stores():
   return {"stores": stores}