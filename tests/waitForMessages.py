import paho.mqtt.client as mqtt
import MyTopicDetails as mtd

# get current topic
myTopic = mtd.returnTopic()

# Subscriber
"""
here connect to redis, create timeseries
"""

def on_connect(client, userdata, flags, rc):
#   print("Connected with result code "+str(rc))
  client.subscribe(myTopic)

def on_message(client, userdata, msg):
  print(msg.topic ,  msg.payload.decode())
  """
  send to redis
  name of ts is AudioIOT
  """
  

client = mqtt.Client()
client.connect("test.mosquitto.org",1883,60)

client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()