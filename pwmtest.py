#!/usr/bin/env python3
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# 1) Initialize I2C bus (busio.I2C uses board.SCL/board.SDA by default)
i2c = busio.I2C(board.SCL, board.SDA)

# 2) Create PCA9685 object at address 0x40
pca = PCA9685(i2c, address=0x40)

# 3) Set the PWM frequency (e.g., 50 Hz for servos; 
#    you can go up to ~1.6 kHz if you want faster switching)
pca.frequency = 50

# 4) Define a helper to convert duty cycle percentage → 16-bit value
def set_duty(channel, percent):
    """
    channel: 0–15
    percent: 0.0 to 100.0 (percentage of full on)
    """
    if not (0 <= percent <= 100):
        raise ValueError("percent must be between 0.0 and 100.0")
    # PCA9685 uses 12-bit internally but busdevice scales it to 16-bit.
    # 0 = always off, 0xFFFF = always on. Linear interpolation for duty.
    duty_16bit = int(percent * 65535 / 100)
    pca.channels[channel].duty_cycle = duty_16bit

# 5) Example: sweep channel 0 from 0% → 100% → 0%
try:
    while True:
        # Ramp from 0 → 100 in 5% steps
        for dc in range(0, 101, 5):
            set_duty(0, dc)
            print(f"Channel 0 duty: {dc}%")
            time.sleep(0.2)
        # Ramp back down
        for dc in range(100, -1, -5):
            set_duty(0, dc)
            print(f"Channel 0 duty: {dc}%")
            time.sleep(0.2)
except KeyboardInterrupt:
    print("Exiting...")

# 6) Clean up: set all channels to 0%
for ch in range(16):
    pca.channels[ch].duty_cycle = 0

# 7) Deinitialize PCA (optional)
pca.deinit()
