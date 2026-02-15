from smbus2 import SMBus
import time

MPU_ADDR = 0x68
bus = SMBus(1)

# Wake MPU6050
bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)
time.sleep(0.2)

# Now read
high = bus.read_byte_data(MPU_ADDR, 0x3B)
low  = bus.read_byte_data(MPU_ADDR, 0x3C)

accel_x = (high << 8) | low
print(accel_x)
