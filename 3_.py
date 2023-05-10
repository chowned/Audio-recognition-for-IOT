import paho.mqtt.client as mqtt
import redis
import MyTopicDetails as mtd
from time import time



# get current topic
myTopic = mtd.returnTopic()

# Subscriber
"""
here connect to redis, create timeseries
"""

REDIS_HOST = '****'
REDIS_PORT = 152
REDIS_USERNAME = '***'
REDIS_PASSWORD = '*******'
timeseries_name = "AudioIOT"

redis_client = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    username=REDIS_USERNAME, 
    password=REDIS_PASSWORD)
is_connected = redis_client.ping()

#redis_client.flushdb()
try:
  redis_client.ts().create(timeseries_name, chunk_size=128)
except redis.ResponseError:
  pass

def on_connect(client, userdata, flags, rc):
#   print("Connected with result code "+str(rc))
  client.subscribe(myTopic)

def on_message(client, userdata, msg):
  print(msg.topic ,  msg.payload.decode())
  """
  send to redis
  name of ts is AudioIOT
  """
  timestamp=time()
  try:
    redis_client.ts().add(timeseries_name,timestamp,msg.payload.decode())
  except redis.exceptions.ResponseError as e:
    pass




client = mqtt.Client()
client.connect("test.mosquitto.org",1883,60)

client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()