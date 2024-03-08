from fastapi import FastAPI, HTTPException,Request, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
from ultralytics import YOLO
import json
import cx_Oracle
import datetime
import requests
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from requests.auth import HTTPBasicAuth
from a2wsgi import ASGIMiddleware

with open('.\\Appsettings.json') as f:
    appsetting = json.load(f)


model_path = appsetting["AppSettings"]["ModelPath"]
labels = ['DTM Fiber Ek Kutusu', 'ERAT Fiber Ek Kutusu','DTM Fiber Ek Kablolu', 'ERAT Fiber Ek Kablolu']


class Database():

    def __init__(self, username, password, host, port, database_name):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database_name = database_name


class DatabaseConnection():

    def __init__(self, appsetting_prod):
        self.appsetting_prod = appsetting_prod

    def returnDatabaseConnect(self, database):
        connection_list = self.appsetting_prod["ConnectionStrings"][database].split(
            ";")
        db = Database(0, 0, 0, 0, 0)
        db.username = connection_list[0].split("=")[1]
        db.password = connection_list[1].split("=")[1]
        db.host = connection_list[2].split(" ")[11].split(")")[0]
        db.port = connection_list[2].split(" ")[13][0:4]
        db.database_name = connection_list[2].split(" ")[-1][:-3]
        return db
    
    
class Item(BaseModel):
    base64_image: str
    file_path: str
    user_id: int
    project_area_id: int
    layer_id: int
    object_id: int

class deneme(BaseModel):
    fileName: str
    fileData: str

class Log:
    def __init__(self,state,start_time,end_time,host,client_ip,base64,file_path,IsDetect,user_id,project_area_id,layer_id,object_id, results):
        self.state = state
        self.start_time = start_time
        self.end_time = end_time
        self.host = host
        self.client_ip = client_ip
        self.base64 = base64
        self.file_path =file_path
        self.IsDetect = IsDetect
        self.user_id = user_id
        self.project_area_id = project_area_id
        self.layer_id = layer_id
        self.object_id = object_id
        self.results =results
    



app = FastAPI()
#app.add_middleware(HTTPSRedirectMiddleware)  

@app.get("/")
def read_root(request: Request):
    client_ip = request.headers.get("Host")
    return {"Hello": client_ip}

@app.post("/predict")
async def predict(item: Item,request: Request, backgroundTasks: BackgroundTasks):
    start_time=datetime.datetime.now()
    client_ip = request.client.host
    host = request.headers.get("Host")
    IsDetect=0
    state=0
    response_list=[]
    label="0"
    try:
        model = YOLO(model_path)
        image_data = base64.b64decode(item.base64_image)
        image = Image.open(BytesIO(image_data))
        #image = image.resize((640,640)) #bunu kapat
        results = model.predict(image, conf=0.8) 
        for result in results:
            boxes = result.boxes.cpu().numpy()
        for i in boxes:
            label = labels[int(i.cls)]
            conf = i.conf[0]
            x1 = int(i.xyxy[0][0])
            y1 = int(i.xyxy[0][1])
            x2 = int(i.xyxy[0][2])
            y2 = int(i.xyxy[0][3])
            bbox_points=[x1, y1, x2, y2]
            #boxes_location = i.boxes[0][0:4].tolist()
            response = {"label":label, "conf": str(conf), "boxes": bbox_points}
            response_list.append(response)
        if label != "0":
            IsDetect=1
        state=1
        end_time=datetime.datetime.now()
        if len(response_list)>0:
            return JSONResponse(status_code=200,content=response_list)

        else:
            return JSONResponse(status_code=209,content={"info":"Non-detected"})
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")



#app = ASGIMiddleware(app)
""" 
if __name__=="__main__":
    uvicorn.run("my_flask:app", host='127.0.0.1',port=8996) #local de yayınlamak için
    #uvicorn.run("app:app", host='0.0.0.0',port=8996) #IIS de çalışan sürüm 
    #uvicorn.run("app:app", host='0.0.0.0',port=8996, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile) #yayınlarken reload kaldır. ssl ile yayınlamak için 
"""
