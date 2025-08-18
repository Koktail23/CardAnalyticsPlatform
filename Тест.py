from clickhouse_driver import Client

client = Client(
    host='localhost',
    port=9000,
    user='analyst',
    password='your_secure_password_here',
    database='card_analytics'
)
print(client.execute('SELECT 1'))