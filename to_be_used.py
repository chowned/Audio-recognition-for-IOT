#redis args
parser.add_argument('--host', default='redis-18608.c250.eu-central-1-1.ec2.cloud.redislabs.com', type=str, help="Default host change for others")
parser.add_argument('--port', default=18608, type=int, help="Default port change for others")
parser.add_argument('--user', default='default', type=str, help="Default user change for others")
parser.add_argument('--password', default='SKVpCpkigmS5xrgsjdhG6TH8N7adlmIB', type=str, help="Default password change for others")
parser.add_argument('--flushDB', default=0, type=int, help="Set 1 to flush all database. Default is 0")


# Connect to Redis
# redis_host, redis_port, REDIS_USERNAME, REDIS_PASSWORD = mc.getMyConnectionDetails()

redis_host     = args.host
redis_port     = args.port
REDIS_USERNAME = args.user
REDIS_PASSWORD = args.password

redis_client = redis.Redis(host=redis_host, port=redis_port, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

mac_address = hex(uuid.getnode())

bucket_1d_in_ms=86400000
one_mb_time_in_ms = 655359000
five_mb_time_in_ms = 3276799000


if args.flushDB == 1:
    try:
        print("Flusing DB")
        redis_client.flushall()
    except redis.ResponseError:
        print("Cannot flush")
        pass
try:
    prefix_TS = "prediction"
    redis_client.ts().create('{prefix_TS}:CL', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:AM', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:DL', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:IV', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:DV', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:IH', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:DH', chunk_size=128, retention=five_mb_time_in_ms)
    redis_client.ts().create('{prefix_TS}:NN', chunk_size=128, retention=five_mb_time_in_ms)
except redis.ResponseError:
    print("Cannot create some TimeSeries as they already exist")
    pass