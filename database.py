from orator import DatabaseManager
config = {
    'mysql': {
        'driver': 'mysql',
        'host': 'omneky-live-backup.mysql.database.azure.com',
        'database': 'wave',
        'user': 'roothikariwave@omneky-live-backup',
        'password': 'Hikari@#wave@root',
        'prefix': ''
    }
}
# config = {
#     'mysql': {
#         'driver': 'mysql',
#         'host': '3.14.121.99',
#         'database': 'wave',
#         'user': 'root',
#         'password': 'zaqxsw123',
#         'prefix': ''
#     }
# }
db = DatabaseManager(config)