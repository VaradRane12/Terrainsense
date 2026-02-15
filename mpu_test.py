from mpu6050 import mpu6050
import time

sensor = mpu6050(0x68)

while True:
    accel = sensor.get_accel_data()
    gyro = sensor.get_gyro_data()
    print("Accel:", accel, "Gyro:", gyro)
    time.sleep(0.5)

