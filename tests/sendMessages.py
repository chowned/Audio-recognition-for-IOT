import paho.mqtt.client as mqtt
import MyTopicDetails as mtd

#get current topic
myTopic = mtd.returnTopic()

# Publisher

client = mqtt.Client()
client.connect("test.mosquitto.org",1883,60)
client.publish(myTopic, 7)
client.disconnect()